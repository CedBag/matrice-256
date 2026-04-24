# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
Tests unitaires — Semaine 1
Projet bokono / Matrice-256

Valide les fondations : encodeur Fâ et matrice 16×16.
Exécuter avec : python -m pytest tests/test_week1.py -v
"""

import pytest
import sys
import os

# Ajout du chemin racine pour les imports
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bokono.core.fa_encoder import (
    kpoli_index, kpoli_name, encode_du, decode_du,
    du_from_kpoli, value_to_du, get_all_meji,
    binary_to_marks, marks_to_binary, KPOLI_16
)
from bokono.core.fa_matrix import FaMatrix
import numpy as np


# ─── T6.1 : Les 256 du sont présents ────────────────────────────────────────
class TestKnowledgeBase:

    def test_256_du_present(self):
        """Tous les 256 du doivent être dans la base de connaissance."""
        from bokono.core.fa_encoder import _get_kb
        kb = _get_kb()
        assert len(kb) == 256, f"Attendu 256, trouvé {len(kb)}"

    def test_all_du_have_id(self):
        """Chaque du doit avoir un id unique de 0 à 255."""
        from bokono.core.fa_encoder import _get_kb
        kb = _get_kb()
        ids = [d["id"] for d in kb]
        assert ids == list(range(256)), "Les IDs ne sont pas séquentiels de 0 à 255"

    def test_all_du_have_binary_code(self):
        """Chaque du doit avoir un code binaire de 8 bits."""
        from bokono.core.fa_encoder import _get_kb
        kb = _get_kb()
        for d in kb:
            code = d.get("code_binaire", "")
            assert len(code) == 8, f"Du #{d['id']} : code binaire invalide '{code}'"
            assert all(c in "01" for c in code), f"Du #{d['id']} : caractères non binaires"

    def test_all_du_have_kpoli(self):
        """Chaque du doit avoir des Kpoli gauche et droit définis."""
        from bokono.core.fa_encoder import _get_kb
        kb = _get_kb()
        for d in kb:
            assert d.get("kpoli_gauche"), f"Du #{d['id']} : kpoli_gauche manquant"
            assert d.get("kpoli_droit"), f"Du #{d['id']} : kpoli_droit manquant"


# ─── T6.2 : Encodage/Décodage ────────────────────────────────────────────────
class TestEncoder:

    def test_16_kpoli_defined(self):
        """Les 16 Kpoli doivent être correctement définis."""
        assert len(KPOLI_16) == 16, f"Attendu 16 Kpoli, trouvé {len(KPOLI_16)}"

    def test_kpoli_index_gbe(self):
        """GBE est le premier Kpoli (index 0)."""
        assert kpoli_index("GBE") == 0

    def test_kpoli_index_ika(self):
        """IKA est le dernier Kpoli (index 15)."""
        assert kpoli_index("IKA") == 15

    def test_kpoli_name_roundtrip(self):
        """Index → nom → index doit être idempotent."""
        for i in range(16):
            name = kpoli_name(i)
            assert kpoli_index(name) == i, f"Roundtrip échoué pour index {i} → {name}"

    def test_encode_gbe_meji(self):
        """GBE × GBE = 00000000 (premier du)."""
        assert encode_du("GBE", "GBE") == "00000000"

    def test_encode_ika_meji(self):
        """IKA × IKA = 11111111 (dernier du)."""
        assert encode_du("IKA", "IKA") == "11111111"

    def test_encode_gbe_yeku(self):
        """GBE × YEKU = 00000001 (deuxième du)."""
        assert encode_du("GBE", "YEKU") == "00000001"

    def test_decode_zero(self):
        """decode_du(0) doit retourner le premier du."""
        du = decode_du(0)
        assert du["id"] == 0
        assert du["code_binaire"] == "00000000"

    def test_decode_255(self):
        """decode_du(255) doit retourner le dernier du."""
        du = decode_du(255)
        assert du["id"] == 255
        assert du["code_binaire"] == "11111111"

    def test_encode_decode_roundtrip(self):
        """Encode puis décode → même du."""
        for i in range(16):
            for j in range(16):
                left = kpoli_name(i)
                right = kpoli_name(j)
                code = encode_du(left, right)
                du = decode_du(code)
                assert du["kpoli_gauche"] == left, \
                    f"Roundtrip {left}×{right} : kpoli_gauche incorrect"
                assert du["kpoli_droit"] == right, \
                    f"Roundtrip {left}×{right} : kpoli_droit incorrect"

    def test_value_to_du_zero(self):
        """value_to_du(0.0) → premier du."""
        du = value_to_du(0.0)
        assert du["id"] == 0

    def test_value_to_du_one(self):
        """value_to_du(1.0) → dernier du."""
        du = value_to_du(1.0)
        assert du["id"] == 255

    def test_binary_to_marks(self):
        """Conversion binaire → marques visuelles Fâ."""
        result = binary_to_marks("10")
        assert "|" in result
        assert "||" in result

    def test_marks_to_binary(self):
        """Conversion marques visuelles → binaire."""
        code = "00000001"
        marks = binary_to_marks(code)
        back = marks_to_binary(marks)
        assert back == code, f"marks_to_binary({marks}) = {back}, attendu {code}"


# ─── T6.3 : Les 16 Meji ───────────────────────────────────────────────────────
class TestMeji:

    def test_exactly_16_meji(self):
        """Il doit y avoir exactement 16 signes Meji (doubles)."""
        meji = get_all_meji()
        assert len(meji) == 16, f"Attendu 16 Meji, trouvé {len(meji)}"

    def test_meji_are_diagonal(self):
        """Les Meji doivent être sur la diagonale (kpoli_gauche == kpoli_droit)."""
        meji = get_all_meji()
        for m in meji:
            assert m["kpoli_gauche"] == m["kpoli_droit"], \
                f"Meji non diagonal : {m['nom_fon']}"

    def test_gbe_meji_is_first(self):
        """GBE-MEJI doit être le premier du (id=0)."""
        du = du_from_kpoli("GBE", "GBE")
        assert du["est_meji"] is True
        assert du["id"] == 0


# ─── T6.4 : La Matrice ───────────────────────────────────────────────────────
class TestMatrix:
    matrix: "FaMatrix"  # annotation de classe : Pyre2 sait que setup_method l'initialise

    def setup_method(self) -> None:
        self.matrix = FaMatrix()

    def test_matrix_completeness(self):
        """La matrice doit avoir exactement 16×16 = 256 cellules."""
        total = sum(len(row) for row in self.matrix._matrix)
        assert total == 256, f"Matrice incomplète : {total}/256 cellules"

    def test_tensor_shape(self):
        """Le tenseur de poids doit avoir la forme (16, 16, 4)."""
        T = self.matrix.to_weight_tensor()
        assert T.shape == (16, 16, 4), f"Forme incorrecte : {T.shape}"

    def test_tensor_dtype(self):
        """Le tenseur doit être float32."""
        T = self.matrix.to_weight_tensor()
        assert T.dtype == np.float32

    def test_tensor_values_in_range(self):
        """Toutes les valeurs du tenseur doivent être entre 0.0 et 1.0."""
        T = self.matrix.to_weight_tensor()
        assert T.min() >= 0.0, f"Valeur minimale hors limites : {T.min()}"
        assert T.max() <= 1.0, f"Valeur maximale hors limites : {T.max()}"

    def test_meji_diagonal(self):
        """La diagonale de la matrice doit contenir exactement 16 Meji."""
        meji = self.matrix.get_meji_diagonal()
        assert len(meji) == 16
        for m in meji:
            assert m["est_meji"] is True

    def test_polarity_map_shape(self):
        """La carte de polarité doit avoir la forme (16, 16)."""
        P = self.matrix.polarity_map()
        assert P.shape == (16, 16)

    def test_lookup_consistency(self):
        """lookup() doit retourner le même résultat que du_from_kpoli()."""
        d1 = self.matrix.lookup("GBE", "YEKU")
        d2 = du_from_kpoli("GBE", "YEKU")
        assert d1["id"] == d2["id"]
        assert d1["code_binaire"] == d2["code_binaire"]
