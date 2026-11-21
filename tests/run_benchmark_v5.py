# Copyright (c) 2026 Cedoque BAGBONON, Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Matrice-256 (FaNet v4.1) : Suite scientifique complète v5.0 (Ablation + 8 Tests corrigés)
=========================================================================================
Conçue pour exécution sur GPU Kaggle (T4 / P100) ou CPU local (--quick).

Expériences incluses :
  Exp 1 : Étude d'ablation contrôlée (9 variantes, 4 datasets réels, 5-Fold x 5 seeds, Wilcoxon)
  Exp 2 : Robustesse au bruit gaussien (F1, PR-AUC, ROC-AUC, Accuracy sur 5 folds x 5 tirages)
  Exp 3 : Généralisation OOD sans fuite de données (Riz -> Jute et Maïs, négatifs strictement disjoints)
  Exp 4 : Calibration probabiliste clinique (ECE pondérée standard + Brier Score + courbes réelles)
  Exp 5 : Régression (Sklearn Diabetes réel + Prix agricoles synthétiques)
  Exp 6 : Séries temporelles IoT (split chronologique + purge 18h + baseline LSTM standard)
  Exp 7 : Audit MIA Confidentialité vs Utilité (attaques loss & confiance + vraies courbes ROC)
  Exp 8 : Efficacité & Frugalité (paramètres, latence par patient, ONNX Runtime, CodeCarbon)

Sorties sauvegardées dans --out-dir et compressées dans --zip-path (par défaut results_v5.zip).
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import platform
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bokono.ml.torch_network import TorchFaNet


# ---------------------------------------------------------------------------
# Utilitaires métriques, statistiques et modèles de contrôle (iso-paramètres)
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) pondérée standard :
      ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    n = len(y_true)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        count = int(mask.sum())
        if count > 0:
            acc_bin = float(y_true[mask].mean())
            conf_bin = float(y_prob[mask].mean())
            ece += (count / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_clf_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": weighted_ece(y_true, y_prob, n_bins=10),
    }


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) == 1 else float("nan")
        return val, val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Test apparié de Wilcoxon (retourne statistique, p_value)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    valid = np.isfinite(diff)
    diff = diff[valid]
    if len(diff) < 5 or np.allclose(diff, 0.0):
        return float("nan"), 1.0
    try:
        res = stats.wilcoxon(diff)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), 1.0


class IsoMLP(nn.Module):
    """
    Baseline MLP PyTorch iso-paramètres face à TorchFaNet v4.2 :
      Input(n_features) -> Linear(n, 4*n) -> ReLU
      -> Linear(4*n, hidden_dim) -> ReLU
      -> Linear(hidden_dim, hidden_dim) -> ReLU (+ résiduel) -> Dropout -> Linear(hidden_dim, 1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        task: str = "classification",
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.seed = seed
        embed_dim = n_features * 4
        self.in_proj = nn.Sequential(nn.Linear(n_features, embed_dim), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.drop = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        e = self.in_proj(x.to(torch.float32))
        h1 = self.fc1(e)
        h = h1 + self.fc2(h1)
        return self.head(self.drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_logits(x)
        return z if self.task == "regression" else torch.sigmoid(z)

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        lr: float = 0.005,
        batch_size: int = 32,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        pos_weight: float = 1.0,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 15,
    ) -> None:
        device = next(self.parameters()).device
        X_dev = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_dev = torch.tensor(np.asarray(y, dtype=np.float32), device=device).reshape(-1, 1)
        n_samples = X_dev.shape[0]
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        pw_t = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
        loss_fn = (
            nn.BCEWithLogitsLoss(pos_weight=pw_t)
            if self.task == "classification"
            else nn.MSELoss()
        )

        X_v_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device) if X_val is not None else None
        y_v_t = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device).reshape(-1, 1) if y_val is not None else None

        best_val = float("inf")
        best_state = None
        no_imp = 0

        self.train()
        for _ in range(epochs):
            perm = torch.randperm(n_samples, generator=gen, device=device)
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad(set_to_none=True)
                out = self.forward_logits(X_dev[idx])
                loss = loss_fn(out, y_dev[idx])
                loss.backward()
                opt.step()

            if X_v_t is not None and y_v_t is not None and early_stopping_patience > 0:
                self.eval()
                with torch.no_grad():
                    v_loss = float(loss_fn(self.forward_logits(X_v_t), y_v_t).item())
                if v_loss < best_val - 1e-5:
                    best_val = v_loss
                    best_state = copy.deepcopy(self.state_dict())
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= early_stopping_patience:
                        break
                self.train()

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        return self.forward(x_t).cpu().numpy()


class IsoLSTM(nn.Module):
    """Baseline LSTM PyTorch standard iso-paramètres face à FaNet LSTM."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 32,
        bidirectional: bool = True,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.n_features = n_features
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.seed = seed
        embed_dim = n_features * 4
        self.proj = nn.Sequential(nn.Linear(n_features, embed_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(p=dropout)
        dec_in = hidden_dim * 2 if bidirectional else hidden_dim
        self.head = nn.Linear(dec_in, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_seq = x.to(torch.float32).reshape(B, self.seq_len, self.n_features)
        emb = self.proj(x_seq)
        _, (h_n, _) = self.lstm(emb)
        h = torch.cat((h_n[-2], h_n[-1]), dim=-1) if self.bidirectional else h_n[-1]
        return self.head(self.drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        lr: float = 0.005,
        batch_size: int = 64,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        pos_weight: float = 1.0,
        early_stopping_patience: int = 15,
    ) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        device = next(self.parameters()).device
        X_t = torch.tensor(np.asarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True, generator=rng)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        pw_t = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw_t)

        X_v_t = torch.tensor(np.asarray(X_val, dtype=np.float32), device=device) if X_val is not None else None
        y_v_t = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device).reshape(-1, 1) if y_val is not None else None

        best_val = float("inf")
        best_state = None
        no_imp = 0

        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self.forward_logits(xb), yb)
                loss.backward()
                opt.step()

            if X_v_t is not None and y_v_t is not None and early_stopping_patience > 0:
                self.eval()
                with torch.no_grad():
                    v_loss = float(loss_fn(self.forward_logits(X_v_t), y_v_t).item())
                if v_loss < best_val - 1e-5:
                    best_val = v_loss
                    best_state = copy.deepcopy(self.state_dict())
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= early_stopping_patience:
                        break
                self.train()

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        return self.forward(x_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Chargement des jeux de données réels
# ---------------------------------------------------------------------------

def load_all_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Pima Indians Diabetes (données médicales réelles du dépôt)
    pima_path = ROOT_DIR / "data_poc" / "sante_dataset.csv"
    df_pima = pd.read_csv(pima_path)
    X_pima = df_pima.drop(columns=["Outcome"]).values.astype(np.float32)
    y_pima = df_pima["Outcome"].values.astype(int)
    datasets["pima_diabetes"] = (X_pima, y_pima)

    # 2. Breast Cancer Wisconsin (données médicales réelles sklearn)
    bc = load_breast_cancer()
    X_bc = bc.data.astype(np.float32)
    # 1 = malin (classe critique), 0 = bénin
    y_bc = (bc.target == 0).astype(int)
    datasets["breast_cancer"] = (X_bc, y_bc)

    # 3. Smart-Agri : cultures à fort besoin hydrique (13.6% positifs)
    agri_path = ROOT_DIR / "data_poc" / "agri_dataset.csv"
    df_agri = pd.read_csv(agri_path).dropna(axis=1, how="all")
    X_agri = df_agri.drop(columns=["label"]).values.astype(np.float32)
    y_agri_hw = df_agri["label"].isin(["rice", "jute", "coconut"]).astype(int).values
    datasets["agri_high_water"] = (X_agri, y_agri_hw)

    # 4. Smart-Agri : cible historique rice (4.5% positifs)
    y_agri_rice = (df_agri["label"] == "rice").astype(int).values
    datasets["agri_rice"] = (X_agri, y_agri_rice)

    return datasets


def train_and_predict_variant(
    variant: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    device: torch.device,
    epochs: int = 80,
    hidden_dim: int = 32,
    pos_weight: float = 1.0,
) -> tuple[np.ndarray, object]:
    """Entraîne un modèle contrôlé et retourne (probas_test, modèle)."""
    n_features = X_tr.shape[1]

    if variant == "B0_LogReg":
        cw = "balanced" if pos_weight > 1.5 else None
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight=cw, random_state=seed)
        clf.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        return clf.predict_proba(X_te)[:, 1], clf

    if variant == "B1_RF_Reg":
        cw = "balanced" if pos_weight > 1.5 else None
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight=cw, random_state=seed, n_jobs=-1,
        )
        clf.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        return clf.predict_proba(X_te)[:, 1], clf

    if variant == "B2_HistGB":
        clf = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.05, random_state=seed
        )
        clf.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        return clf.predict_proba(X_te)[:, 1], clf

    if variant == "M0_MLP_Iso":
        model = IsoMLP(
            n_features=n_features, hidden_dim=hidden_dim, dropout=0.1,
            task="classification", seed=seed,
        ).to(device)
        model.fit(
            X_tr, y_tr, epochs=epochs, lr=0.005, batch_size=32,
            X_val=X_va, y_val=y_va, pos_weight=pos_weight,
            weight_decay=1e-4, early_stopping_patience=15,
        )
        return model.predict_proba(X_te).ravel(), model

    # Variantes TorchFaNet (toutes iso-paramètres et même boucle d'entraînement)
    fanet_configs = {
        "A1_FaNet_NoLoss":   dict(lambda_fa=0.0, polarity_mode="fa",       init_mode="fa",     output_mode="standard"),
        "A2_FaNet_Const05":  dict(lambda_fa=0.1, polarity_mode="constant", init_mode="fa",     output_mode="standard"),
        "A3_FaNet_Permuted": dict(lambda_fa=0.1, polarity_mode="permuted", init_mode="xavier", output_mode="standard"),
        "A4_FaNet_FaDual":   dict(lambda_fa=0.1, polarity_mode="fa",       init_mode="fa",     output_mode="fa_dual"),
        "F1_FaNet_Full":     dict(lambda_fa=0.1, polarity_mode="fa",       init_mode="fa",     output_mode="standard"),
    }
    cfg = fanet_configs[variant]
    model = TorchFaNet(
        n_features=n_features,
        hidden_dim=hidden_dim,
        n_outputs=1,
        activation="fa_sigma",
        seed=seed,
        dropout=0.1,
        task="classification",
        output_mode=cfg["output_mode"],
        polarity_mode=cfg["polarity_mode"],
        init_mode=cfg["init_mode"],
        per_feature_queries=True,
        auto_normalize=True,
    ).to(device)
    model.fit(
        X_tr, y_tr,
        epochs=epochs, lr=0.005, batch_size=32, optimizer="adamw",
        X_val=X_va, y_val=y_va, verbose=False,
        lambda_fa=cfg["lambda_fa"], pos_weight=pos_weight,
        task="classification", weight_decay=1e-4, early_stopping_patience=15,
    )
    return model.predict_proba(X_te).ravel(), model


# ---------------------------------------------------------------------------
# Expérience 1 : Étude d'ablation complète (5-Fold x 5 seeds + Wilcoxon)
# ---------------------------------------------------------------------------

def run_exp1_ablation(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    device: torch.device,
    n_splits: int,
    seeds: list[int],
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 85)
    print(f"  EXP 1 : ÉTUDE D'ABLATION CONTRÔLÉE ({n_splits}-Fold x {len(seeds)} seeds)")
    print("=" * 85)

    variants = [
        "B0_LogReg", "B1_RF_Reg", "B2_HistGB", "M0_MLP_Iso",
        "A1_FaNet_NoLoss", "A2_FaNet_Const05", "A3_FaNet_Permuted",
        "A4_FaNet_FaDual", "F1_FaNet_Full",
    ]

    raw_rows = []
    for ds_name, (X, y) in datasets.items():
        print(f"\n  [Dataset: {ds_name}] N={len(y)}, d={X.shape[1]}, pos={y.mean()*100:.1f}%")
        # Pour les classes rares (<15%), pos_weight modéré symétrique
        pos_ratio = float(y.mean())
        pw = float(np.sqrt((1.0 - pos_ratio) / max(pos_ratio, 1e-6))) if pos_ratio < 0.2 else 1.0

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (tr_full, te_idx) in enumerate(skf.split(X, y), start=1):
                X_tr_full, y_tr_full = X[tr_full], y[tr_full]
                X_te, y_te = X[te_idx], y[te_idx]

                X_tr, X_va, y_tr, y_va = train_test_split(
                    X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=seed + fold
                )

                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_tr)
                X_va_sc = sc.transform(X_va)
                X_te_sc = sc.transform(X_te)

                for var in variants:
                    t0 = time.perf_counter()
                    probs, _ = train_and_predict_variant(
                        var, X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc,
                        seed=seed * 100 + fold, device=device, epochs=epochs,
                        hidden_dim=32, pos_weight=pw,
                    )
                    elapsed = time.perf_counter() - t0
                    m = compute_clf_metrics(y_te, probs, threshold=0.5)
                    raw_rows.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "fold": fold,
                        "variant": var,
                        "train_time_s": elapsed,
                        **m,
                    })

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(out_dir / "exp1_ablation_raw.csv", index=False)

    # Agrégation (moyenne, std, IC95% bootstrap)
    summary_rows = []
    for (ds_name, var), grp in df_raw.groupby(["dataset", "variant"], sort=False):
        row = {"dataset": ds_name, "variant": var, "n_runs": len(grp)}
        for col in ["f1", "roc_auc", "pr_auc", "accuracy", "recall", "precision", "brier", "ece"]:
            vals = grp[col].tolist()
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals))
            lo, hi = bootstrap_ci(vals, n_boot=500, seed=42)
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "exp1_ablation_summary.csv", index=False)

    # Tests statistiques appariés de Wilcoxon (F1_FaNet_Full vs contrôles)
    stat_rows = []
    controls = ["B0_LogReg", "B1_RF_Reg", "B2_HistGB", "M0_MLP_Iso", "A1_FaNet_NoLoss", "A2_FaNet_Const05", "A3_FaNet_Permuted", "A4_FaNet_FaDual"]
    for ds_name in datasets.keys():
        sub_f1 = df_raw[(df_raw["dataset"] == ds_name) & (df_raw["variant"] == "F1_FaNet_Full")].sort_values(["seed", "fold"])
        for ctrl in controls:
            sub_c = df_raw[(df_raw["dataset"] == ds_name) & (df_raw["variant"] == ctrl)].sort_values(["seed", "fold"])
            for metric in ["f1", "roc_auc", "pr_auc", "brier", "ece"]:
                a = sub_f1[metric].values
                b = sub_c[metric].values
                _, pval = safe_wilcoxon(a, b)
                diff_mean = float(np.nanmean(a - b))
                stat_rows.append({
                    "dataset": ds_name,
                    "comparison": f"F1_FaNet_Full vs {ctrl}",
                    "metric": metric,
                    "mean_diff": diff_mean,
                    "wilcoxon_p": pval,
                    "significant_005": bool(pval < 0.05),
                })
    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(out_dir / "exp1_wilcoxon_tests.csv", index=False)

    # Figure 1 : Comparaison F1 et ROC-AUC sur Pima et Breast Cancer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    for ax, ds_name in zip(axes, ["pima_diabetes", "breast_cancer"]):
        sub = df_summary[df_summary["dataset"] == ds_name]
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["f1_mean"], width=0.35, yerr=sub["f1_std"], capsize=3, label="F1-Score", color="#2563eb")
        ax.bar(x + 0.18, sub["roc_auc_mean"], width=0.35, yerr=sub["roc_auc_std"], capsize=3, label="ROC-AUC", color="#10b981")
        ax.set_title(f"Ablation : {ds_name} ({n_splits}-Fold x {len(seeds)} seeds)")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["variant"], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_ablation.png", bbox_inches="tight")
    plt.close()

    print("  [OK] Exp 1 terminée -> exp1_ablation_summary.csv & exp1_wilcoxon_tests.csv")
    return df_summary, df_stats


# ---------------------------------------------------------------------------
# Expérience 2 : Robustesse au bruit gaussien (F1, PR-AUC, ROC-AUC, Accuracy)
# ---------------------------------------------------------------------------

def run_exp2_noise(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    device: torch.device,
    n_splits: int,
    n_draws: int,
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print(f"  EXP 2 : ROBUSTESSE AU BRUIT ({n_splits}-Fold x {n_draws} tirages, F1 & PR-AUC)")
    print("=" * 85)

    variants = ["B1_RF_Reg", "B2_HistGB", "M0_MLP_Iso", "A1_FaNet_NoLoss", "A3_FaNet_Permuted", "F1_FaNet_Full"]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    rows = []

    for ds_name in ["pima_diabetes", "agri_high_water"]:
        X, y = datasets[ds_name]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (tr_full, te_idx) in enumerate(skf.split(X, y), start=1):
            X_tr_full, y_tr_full = X[tr_full], y[tr_full]
            X_te, y_te = X[te_idx], y[te_idx]
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=42 + fold
            )
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_va_sc = sc.transform(X_va)
            X_te_sc = sc.transform(X_te)

            trained_models = {}
            for var in variants:
                _, model = train_and_predict_variant(
                    var, X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc,
                    seed=100 + fold, device=device, epochs=epochs,
                )
                trained_models[var] = model

            for sigma in noise_levels:
                draws = 1 if sigma == 0.0 else n_draws
                for draw in range(draws):
                    rng = np.random.default_rng(1000 * fold + 100 * draw + int(sigma * 100))
                    X_noisy = X_te_sc if sigma == 0.0 else X_te_sc + rng.normal(0.0, sigma, X_te_sc.shape).astype(np.float32)

                    for var, model in trained_models.items():
                        if hasattr(model, "predict_proba"):
                            out = model.predict_proba(X_noisy)
                            probs = out[:, 1] if (out.ndim == 2 and out.shape[1] == 2) else out.ravel()
                        else:
                            probs = model(torch.tensor(X_noisy, dtype=torch.float32)).cpu().numpy().ravel()
                        m = compute_clf_metrics(y_te, probs)
                        rows.append({
                            "dataset": ds_name,
                            "fold": fold,
                            "noise_sigma": sigma,
                            "draw": draw,
                            "variant": var,
                            **m,
                        })

    df = pd.DataFrame(rows)
    df_agg = df.groupby(["dataset", "noise_sigma", "variant"], as_index=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        acc_mean=("accuracy", "mean"),
    )
    df_agg.to_csv(out_dir / "exp2_noise.csv", index=False)

    # Figure 2 : Courbes F1 vs Bruit
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    for ax, ds_name in zip(axes, ["pima_diabetes", "agri_high_water"]):
        sub = df_agg[df_agg["dataset"] == ds_name]
        for var in variants:
            v = sub[sub["variant"] == var].sort_values("noise_sigma")
            ax.plot(v["noise_sigma"] * 100, v["f1_mean"], marker="o", linewidth=2, label=var)
        ax.set_title(f"Robustesse au bruit (F1-Score) : {ds_name}")
        ax.set_xlabel("Bruit gaussien (% de sigma)")
        ax.set_ylabel("F1-Score")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_noise_robustness.png", bbox_inches="tight")
    plt.close()

    print("  [OK] Exp 2 terminée -> exp2_noise.csv & fig2_noise_robustness.png")
    return df_agg


# ---------------------------------------------------------------------------
# Expérience 3 : Généralisation OOD strictement sans fuite de données
# ---------------------------------------------------------------------------

def run_exp3_ood(
    out_dir: Path,
    device: torch.device,
    seeds: list[int],
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 3 : GÉNÉRALISATION OOD SANS FUITE (Riz -> Jute & Maïs, négatifs disjoints)")
    print("=" * 85)

    agri_path = ROOT_DIR / "data_poc" / "agri_dataset.csv"
    df_agri = pd.read_csv(agri_path).dropna(axis=1, how="all")

    neg_crops = ["chickpea", "kidneybeans", "lentil", "mothbeans"]
    df_rice = df_agri[df_agri["label"] == "rice"].drop(columns=["label"]).values.astype(np.float32)
    df_jute = df_agri[df_agri["label"] == "jute"].drop(columns=["label"]).values.astype(np.float32)
    df_maize = df_agri[df_agri["label"] == "maize"].drop(columns=["label"]).values.astype(np.float32)
    df_neg = df_agri[df_agri["label"].isin(neg_crops)].drop(columns=["label"]).values.astype(np.float32)

    variants = ["B0_LogReg", "B1_RF_Reg", "B2_HistGB", "M0_MLP_Iso", "A1_FaNet_NoLoss", "A3_FaNet_Permuted", "F1_FaNet_Full"]
    rows = []

    for seed in seeds:
        # Split strict des positifs (rice) ET des négatifs AVANT toute construction OOD
        r_tr, r_te = train_test_split(df_rice, test_size=0.30, random_state=seed)
        n_tr, n_te = train_test_split(df_neg, test_size=0.30, random_state=seed)

        X_tr_full = np.vstack([r_tr, n_tr])
        y_tr_full = np.concatenate([np.ones(len(r_tr), dtype=int), np.zeros(len(n_tr), dtype=int)])
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=seed
        )

        # 3 jeux de test strictement disjoints des données d'entraînement
        test_sets = {
            "IN_rice":   (np.vstack([r_te, n_te]),     np.concatenate([np.ones(len(r_te), dtype=int), np.zeros(len(n_te), dtype=int)])),
            "OOD_jute":  (np.vstack([df_jute, n_te]),  np.concatenate([np.ones(len(df_jute), dtype=int), np.zeros(len(n_te), dtype=int)])),
            "OOD_maize": (np.vstack([df_maize, n_te]), np.concatenate([np.ones(len(df_maize), dtype=int), np.zeros(len(n_te), dtype=int)])),
        }

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_va_sc = sc.transform(X_va)

        for var in variants:
            # Entraînement symétrique (pos_weight=1.0 pour tous)
            _, model = train_and_predict_variant(
                var, X_tr_sc, y_tr, X_va_sc, y_va, X_va_sc,
                seed=seed, device=device, epochs=epochs, pos_weight=1.0,
            )
            for domain_name, (X_dom, y_dom) in test_sets.items():
                X_dom_sc = sc.transform(X_dom)
                out = model.predict_proba(X_dom_sc)
                probs = out[:, 1] if (out.ndim == 2 and out.shape[1] == 2) else out.ravel()
                m = compute_clf_metrics(y_dom, probs)
                rows.append({"seed": seed, "domain": domain_name, "variant": var, **m})

    df = pd.DataFrame(rows)
    df_agg = df.groupby(["domain", "variant"], as_index=False).agg(
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        pr_auc_mean=("pr_auc", "mean"),
    )
    df_agg.to_csv(out_dir / "exp3_ood.csv", index=False)
    print("  [OK] Exp 3 terminée -> exp3_ood.csv")
    return df_agg


# ---------------------------------------------------------------------------
# Expérience 4 : Calibration clinique réelle (ECE pondérée + Reliability curves)
# ---------------------------------------------------------------------------

def run_exp4_calibration(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    device: torch.device,
    n_splits: int,
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 4 : CALIBRATION CLINIQUE (ECE pondérée standard & Brier Score)")
    print("=" * 85)

    variants = ["B0_LogReg", "B1_RF_Reg", "B2_HistGB", "M0_MLP_Iso", "A4_FaNet_FaDual", "F1_FaNet_Full"]
    rows = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    for ax, ds_name in zip(axes, ["pima_diabetes", "breast_cancer"]):
        X, y = datasets[ds_name]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = {v: np.zeros(len(y), dtype=np.float64) for v in variants}

        for fold, (tr_full, te_idx) in enumerate(skf.split(X, y), start=1):
            X_tr_full, y_tr_full = X[tr_full], y[tr_full]
            X_te = X[te_idx]
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=42 + fold
            )
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_va_sc = sc.transform(X_va)
            X_te_sc = sc.transform(X_te)

            for var in variants:
                probs, _ = train_and_predict_variant(
                    var, X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc,
                    seed=42 + fold, device=device, epochs=epochs, pos_weight=1.0,
                )
                oof_probs[var][te_idx] = probs

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Calibration parfaite")
        for var in variants:
            p = np.clip(oof_probs[var], 0.0, 1.0)
            brier = float(brier_score_loss(y, p))
            ece_w = weighted_ece(y, p, n_bins=10)
            rows.append({
                "dataset": ds_name,
                "variant": var,
                "brier_score": brier,
                "weighted_ece": ece_w,
                "min_prob": float(p.min()),
                "max_prob": float(p.max()),
            })

            # Courbe de fiabilité réelle (10 bins)
            edges = np.linspace(0.0, 1.0, 11)
            xs, ys = [], []
            for i in range(10):
                mask = (p >= edges[i]) & (p <= edges[i + 1] if i == 9 else p < edges[i + 1])
                if mask.sum() >= 5:
                    xs.append(float(p[mask].mean()))
                    ys.append(float(y[mask].mean()))
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=f"{var} (ECE={ece_w*100:.1f}%)")

        ax.set_title(f"Courbe de fiabilité clinique : {ds_name}")
        ax.set_xlabel("Probabilité prédite moyenne")
        ax.set_ylabel("Fréquence empirique réelle")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_calibration_clinical.png", bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp4_calibration.csv", index=False)
    print("  [OK] Exp 4 terminée -> exp4_calibration.csv & fig3_calibration_clinical.png")
    return df


# ---------------------------------------------------------------------------
# Expérience 5 : Régression (Sklearn Diabetes réel + Prix agricoles synthétiques)
# ---------------------------------------------------------------------------

def run_exp5_regression(
    out_dir: Path,
    device: torch.device,
    n_splits: int,
    seeds: list[int],
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 5 : RÉGRESSION (Diabetes médical réel + Prix agricoles synthétiques)")
    print("=" * 85)

    # 1. Sklearn Diabetes réel
    diab = load_diabetes()
    X_diab = diab.data.astype(np.float32)
    y_diab = diab.target.astype(np.float32)

    # 2. Prix agricoles synthétiques (même générateur que T5 v4.0)
    rng = np.random.default_rng(42)
    N = 2000
    temp = rng.normal(25, 5, N)
    rain = rng.normal(100, 40, N)
    ph = rng.normal(6.5, 1, N)
    n_fert = rng.normal(50, 20, N)
    prix = np.clip(
        500 + np.maximum(0, 100 - rain) * 10 + np.maximum(0, temp - 30) * 15 + rng.normal(0, 50, N),
        100, 2500,
    ).astype(np.float32)
    X_prix = np.column_stack([temp, rain, ph, n_fert]).astype(np.float32)

    reg_datasets = {"sklearn_diabetes_real": (X_diab, y_diab), "agri_price_synthetic": (X_prix, prix)}
    rows = []

    for ds_name, (X, y) in reg_datasets.items():
        for seed in seeds[:3]:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (tr_full, te_idx) in enumerate(kf.split(X), start=1):
                X_tr_full, y_tr_full = X[tr_full], y[tr_full]
                X_te, y_te = X[te_idx], y[te_idx]
                X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full, y_tr_full, test_size=0.15, random_state=seed + fold)

                sc_x = StandardScaler()
                X_tr_sc = sc_x.fit_transform(X_tr)
                X_va_sc = sc_x.transform(X_va)
                X_te_sc = sc_x.transform(X_te)

                y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
                y_tr_n = (y_tr - y_mean) / y_std
                y_va_n = (y_va - y_mean) / y_std

                preds_map = {}

                # Ridge
                ridge = Ridge(alpha=1.0)
                ridge.fit(np.vstack([X_tr_sc, X_va_sc]), np.concatenate([y_tr, y_va]))
                preds_map["B0_Ridge"] = ridge.predict(X_te_sc)

                # Random Forest Regressor
                rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed + fold, n_jobs=-1)
                rf.fit(np.vstack([X_tr_sc, X_va_sc]), np.concatenate([y_tr, y_va]))
                preds_map["B1_RF_Reg"] = rf.predict(X_te_sc)

                # MLP Iso-paramètres
                mlp = IsoMLP(n_features=X.shape[1], hidden_dim=32, dropout=0.05, task="regression", seed=seed + fold).to(device)
                mlp.fit(X_tr_sc, y_tr_n, epochs=epochs, lr=0.005, batch_size=32, X_val=X_va_sc, y_val=y_va_n, early_stopping_patience=15)
                preds_map["M0_MLP_Iso"] = mlp.predict_proba(X_te_sc).ravel() * y_std + y_mean

                # FaNet sans FaLoss (tête linéaire corrigée)
                fn0 = TorchFaNet(
                    n_features=X.shape[1], hidden_dim=32, n_outputs=1, seed=seed + fold,
                    task="regression", output_mode="standard", polarity_mode="fa",
                ).to(device)
                fn0.fit(
                    X_tr_sc, y_tr_n, epochs=epochs, lr=0.005, batch_size=32, optimizer="adamw",
                    X_val=X_va_sc, y_val=y_va_n, verbose=False, lambda_fa=0.0, task="regression",
                    weight_decay=1e-4, early_stopping_patience=15,
                )
                preds_map["A1_FaNet_NoLoss"] = fn0.predict_proba(X_te_sc).ravel() * y_std + y_mean

                # FaNet complet (tête linéaire corrigée + FaLoss modérée)
                fn1 = TorchFaNet(
                    n_features=X.shape[1], hidden_dim=32, n_outputs=1, seed=seed + fold,
                    task="regression", output_mode="standard", polarity_mode="fa",
                ).to(device)
                fn1.fit(
                    X_tr_sc, y_tr_n, epochs=epochs, lr=0.005, batch_size=32, optimizer="adamw",
                    X_val=X_va_sc, y_val=y_va_n, verbose=False, lambda_fa=0.05, task="regression",
                    weight_decay=1e-4, early_stopping_patience=15,
                )
                preds_map["F1_FaNet_Full"] = fn1.predict_proba(X_te_sc).ravel() * y_std + y_mean

                for var, yp in preds_map.items():
                    rows.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "fold": fold,
                        "variant": var,
                        "mae": float(mean_absolute_error(y_te, yp)),
                        "rmse": float(np.sqrt(mean_squared_error(y_te, yp))),
                        "r2": float(r2_score(y_te, yp)),
                    })

    df = pd.DataFrame(rows)
    df_agg = df.groupby(["dataset", "variant"], as_index=False).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    )
    df_agg.to_csv(out_dir / "exp5_regression.csv", index=False)
    print("  [OK] Exp 5 terminée -> exp5_regression.csv")
    return df_agg


# ---------------------------------------------------------------------------
# Expérience 6 : IoT temporel (Split chronologique + Purge 18h + IsoLSTM)
# ---------------------------------------------------------------------------

def run_exp6_iot(
    out_dir: Path,
    device: torch.device,
    seeds: list[int],
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 6 : IoT TEMPOREL (Split chronologique + Purge 18h + Baseline LSTM)")
    print("=" * 85)

    lookback = 12
    horizon = 6
    purge = lookback + horizon  # 18 heures de purge entre splits

    np.random.seed(42)
    hours = 60 * 24
    t = np.arange(hours)
    temp = 25 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1, hours)
    hum = 70 + 20 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 2, hours)
    soil = np.zeros(hours)
    soil[0] = 50.0
    for i in range(1, hours):
        drop = 0.5 + 0.02 * max(0, temp[i] - 25)
        val = soil[i - 1] - drop + np.random.normal(0, 0.5)
        if np.random.rand() < 0.02:
            val += 30.0
        soil[i] = np.clip(val, 0, 80)

    df_iot = pd.DataFrame({"temp": temp, "hum": hum, "soil": soil})
    df_iot["danger"] = (df_iot["soil"] < 15.0).astype(int)

    X_list, y_list = [], []
    feats = ["temp", "hum", "soil"]
    for i in range(len(df_iot) - lookback - horizon):
        X_list.append(df_iot.iloc[i : i + lookback][feats].values.flatten())
        y_list.append(int(df_iot.iloc[i + lookback : i + lookback + horizon]["danger"].max()))
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=int)

    n = len(X)
    s1 = int(n * 0.60)
    s2 = int(n * 0.75)
    X_tr, y_tr = X[: s1 - purge], y[: s1 - purge]
    X_va, y_va = X[s1 : s2 - purge], y[s1 : s2 - purge]
    X_te, y_te = X[s2:], y[s2:]

    # StandardScaler ajusté strictement sur Train brut
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)
    X_te_sc = sc.transform(X_te)

    pw = float(np.sqrt((len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)))

    def best_thr(probs_v: np.ndarray, y_v: np.ndarray) -> float:
        if y_v.sum() == 0:
            return 0.5
        prec, rec, thrs = precision_recall_curve(y_v, probs_v)
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        if len(thrs) == 0:
            return 0.5
        return float(thrs[int(np.argmax(f1s[:-1]))])

    rows = []
    for seed in seeds:
        models_eval = {}

        rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        models_eval["B1_RF_Reg"] = (rf.predict_proba(X_va_sc)[:, 1], rf.predict_proba(X_te_sc)[:, 1])

        hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=seed)
        hgb.fit(X_tr_sc, y_tr)
        models_eval["B2_HistGB"] = (hgb.predict_proba(X_va_sc)[:, 1], hgb.predict_proba(X_te_sc)[:, 1])

        iso_lstm = IsoLSTM(n_features=3, seq_len=lookback, hidden_dim=32, bidirectional=True, dropout=0.1, seed=seed).to(device)
        iso_lstm.fit(X_tr_sc, y_tr, epochs=epochs, lr=0.005, batch_size=64, X_val=X_va_sc, y_val=y_va, pos_weight=pw, early_stopping_patience=15)
        models_eval["M0_BiLSTM_Iso"] = (iso_lstm.predict_proba(X_va_sc).ravel(), iso_lstm.predict_proba(X_te_sc).ravel())

        fn_no = TorchFaNet(
            n_features=3, hidden_dim=32, n_outputs=1, seed=seed,
            seq_len=lookback, bidirectional=True, dropout=0.1,
            task="classification", output_mode="standard", polarity_mode="fa",
        ).to(device)
        fn_no.fit(
            X_tr_sc, y_tr, epochs=epochs, lr=0.005, batch_size=64, optimizer="adamw",
            X_val=X_va_sc, y_val=y_va, verbose=False, lambda_fa=0.0, pos_weight=pw,
            weight_decay=1e-4, early_stopping_patience=15,
        )
        models_eval["A1_FaNet_BiLSTM_NoLoss"] = (fn_no.predict_proba(X_va_sc).ravel(), fn_no.predict_proba(X_te_sc).ravel())

        fn_full = TorchFaNet(
            n_features=3, hidden_dim=32, n_outputs=1, seed=seed,
            seq_len=lookback, bidirectional=True, dropout=0.1,
            task="classification", output_mode="standard", polarity_mode="fa",
        ).to(device)
        fn_full.fit(
            X_tr_sc, y_tr, epochs=epochs, lr=0.005, batch_size=64, optimizer="adamw",
            X_val=X_va_sc, y_val=y_va, verbose=False, lambda_fa=0.1, pos_weight=pw,
            weight_decay=1e-4, early_stopping_patience=15,
        )
        models_eval["F1_FaNet_BiLSTM_Full"] = (fn_full.predict_proba(X_va_sc).ravel(), fn_full.predict_proba(X_te_sc).ravel())

        for var, (p_va, p_te) in models_eval.items():
            thr = best_thr(p_va, y_va)
            m = compute_clf_metrics(y_te, p_te, threshold=thr)
            rows.append({"seed": seed, "variant": var, "val_threshold": thr, **m})

    df = pd.DataFrame(rows)
    df_agg = df.groupby("variant", as_index=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
        roc_auc_mean=("roc_auc", "mean"), pr_auc_mean=("pr_auc", "mean"),
    )
    df_agg.to_csv(out_dir / "exp6_iot.csv", index=False)
    print("  [OK] Exp 6 terminée -> exp6_iot.csv")
    return df_agg


# ---------------------------------------------------------------------------
# Expérience 7 : Audit MIA Confidentialité vs Utilité + Vraies courbes ROC
# ---------------------------------------------------------------------------

def run_exp7_mia(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    device: torch.device,
    seeds: list[int],
    epochs: int,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 7 : AUDIT MIA CONFIDENTIALITÉ vs UTILITÉ (Vraies courbes ROC)")
    print("=" * 85)

    X, y = datasets["pima_diabetes"]
    variants = [
        "RF_Unreg", "B1_RF_Reg", "B0_LogReg", "M0_MLP_Iso",
        "A1_FaNet_NoLoss", "A4_FaNet_FaDual", "F1_FaNet_Full",
    ]
    rows = []
    roc_store: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for s_idx, seed in enumerate(seeds):
        X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.40, stratify=y, random_state=seed)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=seed)

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_va_sc = sc.transform(X_va)
        X_te_sc = sc.transform(X_te)

        # Pour RF_Unreg et sklearn, on évalue membres = X_tr_sc (même ensemble exact d'entraînement)
        for var in variants:
            if var == "RF_Unreg":
                model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1)
                model.fit(X_tr_sc, y_tr)
                p_tr = model.predict_proba(X_tr_sc)[:, 1]
                p_te = model.predict_proba(X_te_sc)[:, 1]
            elif var == "B1_RF_Reg":
                model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=seed, n_jobs=-1)
                model.fit(X_tr_sc, y_tr)
                p_tr = model.predict_proba(X_tr_sc)[:, 1]
                p_te = model.predict_proba(X_te_sc)[:, 1]
            elif var == "B0_LogReg":
                model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                model.fit(X_tr_sc, y_tr)
                p_tr = model.predict_proba(X_tr_sc)[:, 1]
                p_te = model.predict_proba(X_te_sc)[:, 1]
            else:
                p_te, model = train_and_predict_variant(
                    var, X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc,
                    seed=seed, device=device, epochs=epochs, pos_weight=1.0,
                )
                p_tr = model.predict_proba(X_tr_sc).ravel()

            p_tr_c = np.clip(p_tr, 1e-6, 1.0 - 1e-6)
            p_te_c = np.clip(p_te, 1e-6, 1.0 - 1e-6)

            # 1. Utilité réelle sur Test
            util = compute_clf_metrics(y_te, p_te_c)

            # 2. Attaque par perte (Yeom et al., 2018) : score = -BCE_loss (plus grand = plus probable membre)
            loss_tr = -(y_tr * np.log(p_tr_c) + (1 - y_tr) * np.log(1 - p_tr_c))
            loss_te = -(y_te * np.log(p_te_c) + (1 - y_te) * np.log(1 - p_te_c))
            y_mem = np.concatenate([np.ones(len(loss_tr)), np.zeros(len(loss_te))])
            scores_loss = np.concatenate([-loss_tr, -loss_te])
            mia_loss_auc = float(roc_auc_score(y_mem, scores_loss))
            fpr, tpr, _ = roc_curve(y_mem, scores_loss)
            tpr_1pct = float(np.interp(0.01, fpr, tpr))

            # 3. Attaque par confiance maximale (comparaison historique)
            conf_tr = np.maximum(p_tr_c, 1.0 - p_tr_c)
            conf_te = np.maximum(p_te_c, 1.0 - p_te_c)
            mia_conf_auc = float(roc_auc_score(y_mem, np.concatenate([conf_tr, conf_te])))

            rows.append({
                "seed": seed,
                "variant": var,
                "test_roc_auc": util["roc_auc"],
                "test_f1": util["f1"],
                "mia_loss_auc": mia_loss_auc,
                "mia_conf_auc": mia_conf_auc,
                "tpr_at_1pct_fpr": tpr_1pct,
                "conf_gap": float(conf_tr.mean() - conf_te.mean()),
            })

            if s_idx == 0:
                roc_store[var] = (fpr, tpr, mia_loss_auc)

    df = pd.DataFrame(rows)
    df_agg = df.groupby("variant", as_index=False).agg(
        test_roc_auc_mean=("test_roc_auc", "mean"),
        test_f1_mean=("test_f1", "mean"),
        mia_loss_auc_mean=("mia_loss_auc", "mean"),
        mia_loss_auc_std=("mia_loss_auc", "std"),
        mia_conf_auc_mean=("mia_conf_auc", "mean"),
        tpr_1pct_mean=("tpr_at_1pct_fpr", "mean"),
    )
    df_agg.to_csv(out_dir / "exp7_mia.csv", index=False)

    # Figure 4 : VRAIES courbes ROC MIA + compromis Privacy vs Utility
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=180)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Hasard (AUC = 0.50)")
    for var, (fpr, tpr, auc_v) in roc_store.items():
        axes[0].plot(fpr, tpr, linewidth=1.8, label=f"{var} (AUC={auc_v:.3f})")
    axes[0].set_title("Vraies courbes ROC d'attaque MIA (Loss-based, Pima)")
    axes[0].set_xlabel("FPR (Taux de faux positifs)")
    axes[0].set_ylabel("TPR (Taux de vrais positifs)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=7.5, loc="lower right")

    for _, r in df_agg.iterrows():
        axes[1].scatter(r["mia_loss_auc_mean"], r["test_roc_auc_mean"], s=90)
        axes[1].annotate(
            r["variant"], (r["mia_loss_auc_mean"], r["test_roc_auc_mean"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    axes[1].axvline(0.55, color="#10b981", linestyle="--", alpha=0.6, label="Zone faible fuite (AUC < 0.55)")
    axes[1].set_title("Compromis Confidentialité (MIA) vs Utilité (Test ROC-AUC)")
    axes[1].set_xlabel("MIA Loss-Attack AUC (plus proche de 0.50 = plus privé)")
    axes[1].set_ylabel("Utilité : Test ROC-AUC (plus haut = meilleur)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_mia_privacy_utility.png", bbox_inches="tight")
    plt.close()

    print("  [OK] Exp 7 terminée -> exp7_mia.csv & fig4_mia_privacy_utility.png")
    return df_agg


# ---------------------------------------------------------------------------
# Expérience 8 : Efficacité, Latence par patient, Export ONNX & CodeCarbon
# ---------------------------------------------------------------------------

def run_exp8_efficiency(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    device: torch.device,
    epochs: int,
    n_infer_reps: int = 500,
) -> pd.DataFrame:
    print("\n" + "=" * 85)
    print("  EXP 8 : EFFICACITÉ, LATENCE PAR PATIENT, ONNX RUNTIME & CODECARBON")
    print("=" * 85)

    X, y = datasets["pima_diabetes"]
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr_full, y_tr_full, test_size=0.15, stratify=y_tr_full, random_state=42)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr).astype(np.float32)
    X_va_sc = sc.transform(X_va).astype(np.float32)
    X_te_sc = sc.transform(X_te).astype(np.float32)
    n_patients_total = n_infer_reps * len(X_te_sc)

    has_codecarbon = False
    try:
        from codecarbon import EmissionsTracker
        has_codecarbon = True
    except ImportError:
        EmissionsTracker = None

    def measure_fn(label: str, fn):
        kwh = float("nan")
        if has_codecarbon and EmissionsTracker is not None:
            em_dir = out_dir / "codecarbon_logs"
            em_dir.mkdir(exist_ok=True)
            tracker = EmissionsTracker(
                project_name=label, output_dir=str(em_dir),
                save_to_file=True, log_level="error", allow_multiple_runs=True,
                measure_power_secs=1,
            )
            tracker.start()
            t0 = time.perf_counter()
            res = fn()
            dt = time.perf_counter() - t0
            tracker.stop()
            csv_p = em_dir / "emissions.csv"
            if csv_p.exists():
                try:
                    df_e = pd.read_csv(csv_p)
                    if len(df_e) > 0:
                        kwh = float(df_e["energy_consumed"].iloc[-1])
                except Exception:
                    pass
            return dt, kwh, res
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        return dt, kwh, res

    rows = []

    # 1. RF Régularisé
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=1)
    t_tr, kwh_tr, _ = measure_fn("eff_rf_train", lambda: rf.fit(X_tr_sc, y_tr))
    t_inf, kwh_inf, _ = measure_fn("eff_rf_infer", lambda: [rf.predict_proba(X_te_sc) for _ in range(n_infer_reps)])
    f1_rf = float(f1_score(y_te, rf.predict(X_te_sc), zero_division=0))
    rows.append({
        "model": "B1_RF_Reg", "params": -1, "train_s": t_tr,
        "infer_ms_per_1k": (t_inf / n_patients_total) * 1e6,
        "nWh_per_patient": (kwh_inf * 1e12 / n_patients_total) if np.isfinite(kwh_inf) else float("nan"),
        "train_uWh": (kwh_tr * 1e9) if np.isfinite(kwh_tr) else float("nan"),
        "f1": f1_rf,
    })

    # 2. HistGB
    hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=42)
    t_tr, kwh_tr, _ = measure_fn("eff_hgb_train", lambda: hgb.fit(X_tr_sc, y_tr))
    t_inf, kwh_inf, _ = measure_fn("eff_hgb_infer", lambda: [hgb.predict_proba(X_te_sc) for _ in range(n_infer_reps)])
    f1_hgb = float(f1_score(y_te, hgb.predict(X_te_sc), zero_division=0))
    rows.append({
        "model": "B2_HistGB", "params": -1, "train_s": t_tr,
        "infer_ms_per_1k": (t_inf / n_patients_total) * 1e6,
        "nWh_per_patient": (kwh_inf * 1e12 / n_patients_total) if np.isfinite(kwh_inf) else float("nan"),
        "train_uWh": (kwh_tr * 1e9) if np.isfinite(kwh_tr) else float("nan"),
        "f1": f1_hgb,
    })

    # 3. MLP Iso-paramètres (CPU inference propre avec tenseur pré-alloué)
    mlp = IsoMLP(n_features=X.shape[1], hidden_dim=32, dropout=0.1, seed=42).to(device)
    t_tr, kwh_tr, _ = measure_fn(
        "eff_mlp_train",
        lambda: mlp.fit(X_tr_sc, y_tr, epochs=epochs, lr=0.005, batch_size=32, X_val=X_va_sc, y_val=y_va),
    )
    mlp_cpu = mlp.to("cpu").eval()
    x_te_tensor = torch.tensor(X_te_sc, dtype=torch.float32)

    def _infer_mlp():
        with torch.no_grad():
            for _ in range(n_infer_reps):
                _ = mlp_cpu(x_te_tensor)

    t_inf, kwh_inf, _ = measure_fn("eff_mlp_infer", _infer_mlp)
    f1_mlp = float(f1_score(y_te, (mlp_cpu.predict_proba(X_te_sc).ravel() >= 0.5).astype(int), zero_division=0))
    rows.append({
        "model": "M0_MLP_Iso", "params": mlp_cpu.count_parameters(), "train_s": t_tr,
        "infer_ms_per_1k": (t_inf / n_patients_total) * 1e6,
        "nWh_per_patient": (kwh_inf * 1e12 / n_patients_total) if np.isfinite(kwh_inf) else float("nan"),
        "train_uWh": (kwh_tr * 1e9) if np.isfinite(kwh_tr) else float("nan"),
        "f1": f1_mlp,
    })

    # 4. FaNet v4.1 PyTorch (inférence CPU sur tenseur pré-alloué)
    fanet = TorchFaNet(
        n_features=X.shape[1], hidden_dim=32, n_outputs=1, seed=42,
        dropout=0.1, task="classification", output_mode="standard", polarity_mode="fa",
    ).to(device)
    t_tr, kwh_tr, _ = measure_fn(
        "eff_fanet_train",
        lambda: fanet.fit(
            X_tr_sc, y_tr, epochs=epochs, lr=0.005, batch_size=32, optimizer="adamw",
            X_val=X_va_sc, y_val=y_va, verbose=False, lambda_fa=0.1, early_stopping_patience=15,
        ),
    )
    fanet_cpu = fanet.to("cpu").eval()

    def _infer_fanet():
        with torch.no_grad():
            for _ in range(n_infer_reps):
                _ = fanet_cpu(x_te_tensor)

    t_inf, kwh_inf, _ = measure_fn("eff_fanet_infer", _infer_fanet)
    f1_fa = float(f1_score(y_te, (fanet_cpu.predict_proba(X_te_sc).ravel() >= 0.5).astype(int), zero_division=0))
    rows.append({
        "model": "F1_FaNet_PyTorch", "params": fanet_cpu.count_parameters(), "train_s": t_tr,
        "infer_ms_per_1k": (t_inf / n_patients_total) * 1e6,
        "nWh_per_patient": (kwh_inf * 1e12 / n_patients_total) if np.isfinite(kwh_inf) else float("nan"),
        "train_uWh": (kwh_tr * 1e9) if np.isfinite(kwh_tr) else float("nan"),
        "f1": f1_fa,
    })

    # 5. FaNet v4.2 ONNX Runtime (optimisé 1 thread CPU pour petits batchs Edge)
    try:
        import onnxruntime as ort
        onnx_path = str(out_dir / "fanet_v42_pima.onnx")
        fanet_cpu.export_onnx(onnx_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        t_inf_ox, kwh_inf_ox, _ = measure_fn(
            "eff_fanet_onnx",
            lambda: [sess.run(None, {inp_name: X_te_sc}) for _ in range(n_infer_reps)],
        )
        onnx_out = sess.run(None, {inp_name: X_te_sc})[0].ravel()
        f1_ox = float(f1_score(y_te, (onnx_out >= 0.5).astype(int), zero_division=0))
        rows.append({
            "model": "F1_FaNet_ONNX", "params": fanet_cpu.count_parameters(), "train_s": t_tr,
            "infer_ms_per_1k": (t_inf_ox / n_patients_total) * 1e6,
            "nWh_per_patient": (kwh_inf_ox * 1e12 / n_patients_total) if np.isfinite(kwh_inf_ox) else float("nan"),
            "train_uWh": (kwh_tr * 1e9) if np.isfinite(kwh_tr) else float("nan"),
            "f1": f1_ox,
        })
    except Exception as e:
        print(f"  [INFO] Benchmark ONNX ignoré ({e})")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp8_efficiency.csv", index=False)

    # Figure 5 : Comparaison honnête de TOUS les modèles
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    axes[0].bar(df["model"], df["infer_ms_per_1k"], color="#2563eb")
    axes[0].set_title("Latence d'inférence CPU (ms / 1 000 patients)")
    axes[0].set_ylabel("Millisecondes (plus bas = meilleur)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    if df["nWh_per_patient"].notna().any():
        axes[1].bar(df["model"], df["nWh_per_patient"], color="#10b981")
        axes[1].set_title("Énergie d'inférence (nWh par patient, CodeCarbon)")
        axes[1].set_ylabel("nWh / patient (plus bas = meilleur)")
    else:
        axes[1].bar(df["model"], df["train_s"], color="#f59e0b")
        axes[1].set_title("Temps d'entraînement (secondes)")
        axes[1].set_ylabel("Secondes")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_efficiency_all.png", bbox_inches="tight")
    plt.close()

    print("  [OK] Exp 8 terminée -> exp8_efficiency.csv & fig5_efficiency_all.png")
    return df


# ---------------------------------------------------------------------------
# Génération du rapport Markdown et compression ZIP
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    """Convertit un DataFrame en tableau Markdown sans dépendance externe (tabulate)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def write_summary_report(
    out_dir: Path,
    meta: dict,
    exp1: pd.DataFrame,
    exp1_stats: pd.DataFrame,
    exp2: pd.DataFrame,
    exp3: pd.DataFrame,
    exp4: pd.DataFrame,
    exp5: pd.DataFrame,
    exp6: pd.DataFrame,
    exp7: pd.DataFrame,
    exp8: pd.DataFrame,
) -> None:
    lines = [
        "# Rapport de benchmarks scientifiques v5.0 : Matrice-256 (FaNet v4.2)",
        "",
        f"- Date : {meta['timestamp']}",
        f"- Plateforme : {meta['platform']}",
        f"- PyTorch : {meta['torch_version']} | Device : {meta['device']} ({meta['gpu_name']})",
        f"- Durée totale : {meta['total_seconds']:.1f} s",
        f"- Configuration : {meta['n_splits']}-Fold x {meta['n_seeds']} seeds, {meta['epochs']} époques max (early stopping)",
        "",
        "## Exp 1 : Étude d'ablation (F1, ROC-AUC, PR-AUC, Brier, ECE pondérée)",
        "",
        df_to_markdown(exp1[["dataset", "variant", "f1_mean", "f1_std", "roc_auc_mean", "pr_auc_mean", "brier_mean", "ece_mean"]]),
        "",
        "## Exp 1b : Tests appariés de Wilcoxon (F1_FaNet_Full vs contrôles)",
        "",
        df_to_markdown(exp1_stats),
        "",
        "## Exp 2 : Robustesse au bruit gaussien",
        "",
        df_to_markdown(exp2),
        "",
        "## Exp 3 : Généralisation OOD sans fuite",
        "",
        df_to_markdown(exp3),
        "",
        "## Exp 4 : Calibration clinique (ECE pondérée standard)",
        "",
        df_to_markdown(exp4),
        "",
        "## Exp 5 : Régression (Diabetes réel & Prix synthétiques)",
        "",
        df_to_markdown(exp5),
        "",
        "## Exp 6 : Séries temporelles IoT (Purge 18h + BiLSTM baseline)",
        "",
        df_to_markdown(exp6),
        "",
        "## Exp 7 : Audit MIA Confidentialité vs Utilité",
        "",
        df_to_markdown(exp7),
        "",
        "## Exp 8 : Efficacité, Latence par patient & ONNX",
        "",
        df_to_markdown(exp8),
        "",
    ]
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrice-256 v5.0 Scientific Benchmark & Ablation Suite")
    parser.add_argument("--out-dir", type=str, default="results_v5", help="Dossier de sortie des résultats")
    parser.add_argument("--zip-path", type=str, default="results_v5.zip", help="Archive ZIP finale à télécharger")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quick", action="store_true", help="Mode rapide (2 folds x 2 seeds, 25 époques)")
    args = parser.parse_args()

    t_global = time.perf_counter()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_splits = 2 if args.quick else 5
    seeds = [42, 101] if args.quick else [42, 101, 202, 303, 404]
    epochs = 25 if args.quick else 80
    n_noise_draws = 2 if args.quick else 5
    n_infer_reps = 100 if args.quick else 500

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  MATRICE-256 (FaNet v4.2) : SUITE SCIENTIFIQUE v5.0")
    print(f"  Device : {device} ({gpu_name}) | Mode : {'QUICK' if args.quick else 'FULL'}")
    print(f"  Sortie : {out_dir}")
    print("=" * 85)

    datasets = load_all_datasets()

    exp1, exp1_stats = run_exp1_ablation(datasets, out_dir, device, n_splits, seeds, epochs)
    exp2 = run_exp2_noise(datasets, out_dir, device, n_splits, n_noise_draws, epochs)
    exp3 = run_exp3_ood(out_dir, device, seeds, epochs)
    exp4 = run_exp4_calibration(datasets, out_dir, device, n_splits, epochs)
    exp5 = run_exp5_regression(out_dir, device, n_splits, seeds, epochs)
    exp6 = run_exp6_iot(out_dir, device, seeds, epochs)
    exp7 = run_exp7_mia(datasets, out_dir, device, seeds, epochs)
    exp8 = run_exp8_efficiency(datasets, out_dir, device, epochs, n_infer_reps=n_infer_reps)

    total_seconds = time.perf_counter() - t_global
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "quick_mode": args.quick,
        "n_splits": n_splits,
        "n_seeds": len(seeds),
        "epochs": epochs,
        "total_seconds": total_seconds,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        write_summary_report(out_dir, meta, exp1, exp1_stats, exp2, exp3, exp4, exp5, exp6, exp7, exp8)
    except Exception as e:
        print(f"  [WARN] Impossible de générer summary_report.md (tabulate manquant ?): {e}")

    # Création de l'archive ZIP prête au téléchargement depuis Kaggle
    zip_path = Path(args.zip_path).resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(out_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(out_dir.parent))

    print("\n" + "=" * 85)
    print(f"  TERMINÉ EN {total_seconds:.1f} s")
    print(f"  Archive prête à télécharger : {zip_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
