# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.core.fa_matrix — La Matrice 16×16 du Fâ
================================================
Implémentation de la Matrice-256 : la table de référence centrale du projet.

Structure :
  - Axes : 16 Kpoli × 16 Kpoli = 256 cellules
  - Chaque cellule [i][j] contient le du résultant + son vecteur de caractéristiques
  - Ce vecteur est directement utilisable comme couche d'embedding pour FaNet

Le vecteur de caractéristiques d'un du [valeur_decimale / 255, polarite_num, intensite, est_meji]
servira de poids initiaux pour les couches cachées de FaNet (Semaine 3).

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from bokono.core.fa_encoder import (
    KPOLI_16, kpoli_name, kpoli_index, du_from_kpoli, _get_kb
)


# ─── Mapping polarité → valeur numérique ────────────────────────────────────
_POLARITE_NUM = {
    "favorable":  1.0,
    "neutre":     0.5,
    "défavorable": 0.0,
    "non_défini": 0.5,
}


class FaMatrix:
    """
    La Matrice-256 : table de référence 16×16 du Fâ.

    Exemple d'utilisation :
        matrix = FaMatrix()
        result = matrix.lookup("GBE", "YEKU")
        print(result["nom_fon"], result["polarite"])

        # Pour FaNet : obtenir le tenseur de poids 16×16×4
        W = matrix.to_weight_tensor()
        print(W.shape)  # (16, 16, 4)
    """

    def __init__(self):
        self._kb = _get_kb()
        self._matrix: list[list[dict]] = []
        self._tensor: Optional[np.ndarray] = None
        self._build()

    def _build(self) -> None:
        """Construit la matrice 16×16 depuis la base de connaissance."""
        self._matrix = []
        for i in range(16):
            row = []
            for j in range(16):
                du_id = i * 16 + j  # Position dans la KB
                entry = self._kb[du_id].copy()

                # Vecteur de caractéristiques (4 dimensions)
                entry["_vecteur"] = self._compute_vector(entry, i, j)
                row.append(entry)
            self._matrix.append(row)

    def _compute_vector(self, du: dict, i: int, j: int) -> list[float]:
        """
        Calcule le vecteur de caractéristiques d'un du.

        Dimensions :
          [0] valeur_norm  : position dans la matrice, normalisée 0→1
          [1] polarite     : favorable=1.0, neutre=0.5, défavorable=0.0
          [2] intensite    : force du signe (0.0 → 1.0)
          [3] est_meji     : signe double (1.0) ou non (0.0)
        """
        val_norm = du["valeur_decimale"] / 255.0
        polarite = _POLARITE_NUM.get(du.get("polarite", "non_défini"), 0.5)
        intensite = du.get("intensite", 0.5)
        est_meji = 1.0 if du.get("est_meji", False) else 0.0
        return [val_norm, polarite, intensite, est_meji]

    def lookup(self, left: str, right: str) -> dict:
        """
        Retourne le du complet pour une paire de Kpoli.

        Args:
            left:  Nom du Kpoli de gauche (ex: "GBE")
            right: Nom du Kpoli de droite (ex: "YEKU")

        Returns:
            Dictionnaire du du avec son vecteur de caractéristiques

        Exemple:
            >>> m = FaMatrix()
            >>> d = m.lookup("GBE", "YEKU")
            >>> d["polarite"]
            'favorable'
        """
        i = kpoli_index(left)
        j = kpoli_index(right)
        return self._matrix[i][j]

    def lookup_by_id(self, du_id: int) -> dict:
        """Retourne le du par son identifiant 0-255."""
        if not 0 <= du_id <= 255:
            raise ValueError(f"ID hors limites : {du_id}")
        i, j = divmod(du_id, 16)
        return self._matrix[i][j]

    def to_weight_tensor(self) -> np.ndarray:
        """
        Exporte la matrice comme tenseur NumPy (16, 16, 4).

        Chaque cellule [i, j] = vecteur de 4 caractéristiques du du.
        Ce tenseur est la couche d'embedding native de FaNet.

        Returns:
            np.ndarray de forme (16, 16, 4), dtype float32
        """
        if self._tensor is None:
            T = np.zeros((16, 16, 4), dtype=np.float32)
            for i in range(16):
                for j in range(16):
                    T[i, j] = self._matrix[i][j]["_vecteur"]
            self._tensor = T
        return self._tensor

    def polarity_map(self) -> np.ndarray:
        """
        Retourne une matrice 16×16 de polarités numériques.

        Utile pour visualiser la "topologie" du Fâ :
        les zones favorables et défavorables de l'espace des signes.

        Returns:
            np.ndarray de forme (16, 16), valeurs entre 0.0 et 1.0
        """
        P = np.zeros((16, 16), dtype=np.float32)
        for i in range(16):
            for j in range(16):
                P[i, j] = _POLARITE_NUM.get(
                    self._matrix[i][j].get("polarite", "non_défini"), 0.5
                )
        return P

    def get_meji_diagonal(self) -> list[dict]:
        """
        Retourne les 16 signes Meji (la diagonale de la matrice).
        Ce sont les signes les plus puissants du Fâ.
        """
        return [self._matrix[i][i] for i in range(16)]

    def neighbors(self, left: str, right: str, radius: int = 1) -> list[dict]:
        """
        Retourne les du voisins d'un du donné dans la matrice.

        Utilisé pour la logique de "contexte" dans FaNet :
        un signe ne s'interprète pas isolément, mais dans son voisinage.

        Args:
            left:   Kpoli de gauche
            right:  Kpoli de droite
            radius: Rayon de voisinage (1 = voisins immédiats)

        Returns:
            Liste des du voisins
        """
        ci = kpoli_index(left)
        cj = kpoli_index(right)
        r: int = int(radius)
        neg_r: int = 0 - r  # évite l'opérateur unaire - que Pyre2 ne résout pas
        result: list[dict] = []
        for di in range(neg_r, r + 1):
            for dj in range(neg_r, r + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ci + di, cj + dj
                if 0 <= ni <= 15 and 0 <= nj <= 15:
                    result.append(self._matrix[ni][nj])
        return result


    def summary(self) -> None:
        """Affiche un résumé statistique de la matrice."""
        tensor = self.to_weight_tensor()
        pol_map = self.polarity_map()
        print("=" * 60)
        print("  MATRICE-256 — Résumé")
        print("=" * 60)
        print(f"  Dimensions      : 16 × 16 = 256 du")
        print(f"  Tenseur         : {tensor.shape} (float32)")
        print(f"  Polarité moy.   : {pol_map.mean():.3f} (0=déf, 0.5=neutre, 1=fav)")
        print(f"  Polarité max    : {pol_map.max():.3f}")
        print(f"  Polarité min    : {pol_map.min():.3f}")
        meji = self.get_meji_diagonal()
        print(f"  Signes Meji     : {len(meji)} (diagonale)")
        print("=" * 60)
        print("  Diagonale Meji  :")
        for m in meji:
            pol_sym = "✅" if m["polarite"] == "favorable" else (
                "⛔" if m["polarite"] == "défavorable" else "⚪")
            print(f"    [{m['id']:3d}] {m['nom_fon'][:20]:<20s} {pol_sym} {m['polarite']}")
        print("=" * 60)
