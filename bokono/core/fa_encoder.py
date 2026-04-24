# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""
bokono.core.fa_encoder — Encodeur Fâ ↔ Binaire
================================================
Traducteur universel entre la logique du Fâ et la logique binaire moderne.

Le Fâ utilise un système de marques :
  - Marque simple (impaire / ouvert) → 1 (actif)
  - Marque double (paire / fermé)   → 0 (passif)

Les 16 Kpoli s'encodent sur 4 bits (2⁴ = 16).
Les 256 du s'encodent sur 8 bits (2⁸ = 256) = 1 octet.

Auteur : Projet Matrice-256 / bokono
"""

from __future__ import annotations
import json
import os
from typing import Optional, Union
try:
    from typing import TypedDict
except ImportError:  # Python < 3.8 fallback (non applicable ici)
    from typing_extensions import TypedDict  # type: ignore[no-redef]


class KpoliEntry(TypedDict):
    """Structure d'une entrée du référentiel des 16 Kpoli."""
    index: int
    nom: str
    code: str
    alias: list[str]

# ─── Référentiel des 16 Kpoli (ordre canonique Fon) ──────────────────────────
KPOLI_16 = [
    {"index": 0,  "nom": "GBE",      "code": "0000", "alias": ["GBÉ", "GBHE"]},
    {"index": 1,  "nom": "YEKU",     "code": "0001", "alias": ["YEKOU", "YÉKU"]},
    {"index": 2,  "nom": "WOLI",     "code": "0010", "alias": ["WOLI"]},
    {"index": 3,  "nom": "DI",       "code": "0011", "alias": ["ODI"]},
    {"index": 4,  "nom": "ABLA",     "code": "0100", "alias": ["ABLAMEDJI"]},
    {"index": 5,  "nom": "AKLAN",    "code": "0101", "alias": ["AKLAN"]},
    {"index": 6,  "nom": "TRINMI",   "code": "0110", "alias": ["TRINMI"]},
    {"index": 7,  "nom": "GUDA",     "code": "0111", "alias": ["GUDA"]},
    {"index": 8,  "nom": "SA",       "code": "1000", "alias": ["SA"]},
    {"index": 9,  "nom": "KA",       "code": "1001", "alias": ["KA"]},
    {"index": 10, "nom": "TULA",     "code": "1010", "alias": ["TULA"]},
    {"index": 11, "nom": "TURUKPE",  "code": "1011", "alias": ["TURUKPE"]},
    {"index": 12, "nom": "KANA",     "code": "1100", "alias": ["KANA"]},
    {"index": 13, "nom": "OTURA",    "code": "1101", "alias": ["IRETE"]},
    {"index": 14, "nom": "OKANRAN",  "code": "1110", "alias": ["OSE"]},
    {"index": 15, "nom": "IKA",      "code": "1111", "alias": ["OFUN"]},
]

# Index de recherche rapide
_KPOLI_BY_NAME: dict[str, int] = {}
_k: KpoliEntry
for _k in KPOLI_16:  # type: ignore[assignment]
    _KPOLI_BY_NAME[_k["nom"].upper()] = _k["index"]
    for _alias in _k["alias"]:
        _KPOLI_BY_NAME[_alias.upper()] = _k["index"]

# Chemins de la base de connaissance (Fallback de sécurité)
_KB_PRIVATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "fa_knowledge_private.json"
)
_KB_PUBLIC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "fa_knowledge_public.json"
)


def _load_knowledge_base() -> list[dict]:
    """Charge la base de connaissance Fâ (lazy loading avec Fallback privé/public)."""
    path_priv = os.path.normpath(_KB_PRIVATE_PATH)
    path_pub = os.path.normpath(_KB_PUBLIC_PATH)
    
    # 1. Tente de charger la vraie intelligence (Usage Propriétaire Local)
    if os.path.exists(path_priv):
        try:
            with open(path_priv, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Fichier de base de connaissance corrompu (Private) : {e}")

    # 2. Fallback de sécurité (Usage GitHub / Tiers)
    if os.path.exists(path_pub):
        try:
            with open(path_pub, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Fichier de base de connaissance corrompu (Public) : {e}")
            
    raise FileNotFoundError(
        f"Base de connaissance (Publique ou Privée) introuvable dans data/knowledge_base/\n"
    )

_KB: Optional[list[dict]] = None  # Cache en mémoire
_KB_HASH: Optional[str] = None

def _get_kb() -> list[dict]:
    global _KB, _KB_HASH
    if _KB is None:
        import hashlib
        _KB = _load_knowledge_base()
        _KB_HASH = hashlib.sha256(json.dumps(_KB, sort_keys=True).encode("utf-8")).hexdigest()
    return _KB

def get_kb_hash() -> str:
    """Retourne l'empreinte SHA-256 de la base de connaissance active."""
    if _KB_HASH is None:
        _get_kb()
    assert _KB_HASH is not None
    return _KB_HASH


# ─── Fonctions principales ────────────────────────────────────────────────────

def kpoli_index(nom: str) -> int:
    """
    Retourne l'index (0-15) d'un Kpoli par son nom.

    Args:
        nom: Nom du Kpoli (ex: "GBE", "YEKU", "AKLAN")

    Returns:
        Index 0-15

    Raises:
        ValueError si le nom est inconnu.

    Exemple:
        >>> kpoli_index("GBE")
        0
        >>> kpoli_index("YEKU")
        1
    """
    key = nom.strip().upper()
    if key in _KPOLI_BY_NAME:
        return _KPOLI_BY_NAME[key]
    raise ValueError(
        f"Kpoli inconnu : '{nom}'. "
        f"Valeurs valides : {[k['nom'] for k in KPOLI_16]}"
    )


def kpoli_name(index: int) -> str:
    """
    Retourne le nom Fon d'un Kpoli par son index.

    Args:
        index: Entier 0-15

    Returns:
        Nom du Kpoli (ex: "GBE")

    Exemple:
        >>> kpoli_name(0)
        'GBE'
    """
    if not 0 <= index <= 15:
        raise ValueError(f"Index Kpoli invalide : {index}. Attendu : 0-15.")
    return str(KPOLI_16[index]["nom"])  # cast explicite : TypedDict garantit str


def encode_du(left: Union[str, int], right: Union[str, int]) -> str:
    """
    Encode deux Kpoli en un code binaire 8 bits représentant un du.

    Args:
        left:  Kpoli de gauche (nom ou index 0-15)
        right: Kpoli de droite (nom ou index 0-15)

    Returns:
        Code binaire 8 bits (ex: "00000000")

    Exemples:
        >>> encode_du("GBE", "GBE")
        '00000000'
        >>> encode_du("GBE", "YEKU")
        '00000001'
        >>> encode_du(1, 0)
        '00010000'
    """
    li = left if isinstance(left, int) else kpoli_index(left)
    ri = right if isinstance(right, int) else kpoli_index(right)
    if not (0 <= li <= 15 and 0 <= ri <= 15):
        raise ValueError(f"Index hors limites : left={li}, right={ri}")
    return f"{li:04b}{ri:04b}"


def decode_du(code: Union[str, int]) -> dict:
    """
    Décode un code binaire (ou entier 0-255) en entrée de la base de connaissance.

    Args:
        code: Code binaire 8 bits (ex: "00000001") ou entier 0-255

    Returns:
        Dictionnaire complet du du (depuis fa_knowledge.json)

    Exemples:
        >>> decode_du("00000000")["nom_fon"]
        'GBE MEDJI'
        >>> decode_du(1)["numero"]
        2
    """
    if isinstance(code, int):
        idx = code
    else:
        idx = int(code, 2)

    if not 0 <= idx <= 255:
        raise ValueError(f"Code hors limites : {code}. Attendu : 0-255.")

    kb = _get_kb()
    return kb[idx]


def du_from_kpoli(left: Union[str, int], right: Union[str, int]) -> dict:
    """
    Retourne le du complet à partir de deux Kpoli.

    C'est la fonction principale du système : simule le résultat d'un tirage Fâ.

    Args:
        left:  Kpoli de gauche
        right: Kpoli de droite

    Returns:
        Dictionnaire complet du du

    Exemple:
        >>> d = du_from_kpoli("GBE", "YEKU")
        >>> print(d["nom_fon"], d["polarite"])
    """
    code = encode_du(left, right)
    return decode_du(code)


def value_to_du(value: float, domain: float = 1.0) -> dict:
    """
    Convertit une valeur numérique (float) en du correspondant.

    Utilisé par le moteur ML pour "lire" une sortie de réseau de neurones
    à travers la grille de lecture du Fâ.

    Args:
        value:  Valeur numérique (typiquement entre 0.0 et domain)
        domain: Valeur maximale (défaut: 1.0)

    Returns:
        Dictionnaire du du le plus proche

    Exemple:
        >>> d = value_to_du(0.5)
        >>> print(d["id"], d["nom_canonique"])
    """
    normalized = max(0.0, min(1.0, value / domain))
    idx = int(normalized * 255)
    return decode_du(idx)

def value_to_algorithmic_state(value: float, domain: float = 1.0) -> dict:
    """
    Interface ML Externe.
    Convertit la valeur en un état mathématique pur (Ontologie Sécurisée).
    Supprime tous les vocabulaires culturels (Mèdji, Gbé) pour l'API.
    """
    raw_du = value_to_du(value, domain)
    # Traduction en ontologie algorithmique universelle
    # On garantit l'anonymisation culturelle
    polarite_str = str(raw_du.get("polarite", "neutre")).upper()
    return {
        "state_id": int(raw_du.get("numero", 0)),
        "binary_matrix": f"{raw_du.get('code_binaire', '00000000')[:4]} x {raw_du.get('code_binaire', '00000000')[4:]}",
        "ontological_state": f"STATE_{raw_du.get('numero', 0):03d}",
        "system_polarity": polarite_str,
        "is_primary_node": bool(raw_du.get("est_meji", False))
    }


def get_all_meji() -> list[dict]:
    """Retourne les 16 signes Meji (doubles — les plus puissants du Fâ)."""
    kb = _get_kb()
    return [d for d in kb if d.get("est_meji", False)]


def binary_to_marks(code: str) -> str:
    """
    Convertit un code binaire en représentation visuelle Fâ.
    1 → "|"  (marque simple, ouvert)
    0 → "||" (marque double, fermé)

    Exemple:
        >>> binary_to_marks("00000001")
        '|| || || || || || || |'
    """
    return " ".join("|" if b == "1" else "||" for b in code)


def marks_to_binary(marks: str) -> str:
    """
    Convertit une représentation visuelle Fâ en code binaire.
    "|"  → 1
    "||" → 0

    Exemple:
        >>> marks_to_binary("|| || || || || || || |")
        '00000001'
    """
    parts = marks.strip().split()
    result = ""
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i] == "|" and parts[i+1] == "|":
            result += "0"
            i += 2
        elif parts[i] == "||":
            result += "0"
            i += 1
        else:
            result += "1"
            i += 1
    return result


# ─── Affichage ────────────────────────────────────────────────────────────────

def print_du(du: dict, verbose: bool = False) -> None:
    """Affiche un du de façon lisible."""
    sep = "─" * 60
    print(sep)
    print(f"  DU #{du['numero']:3d} | {du['nom_fon']}")
    print(f"  Binaire   : {du['code_binaire']}  →  {binary_to_marks(du['code_binaire'])}")
    print(f"  Kpoli     : {du['kpoli_gauche']} × {du['kpoli_droit']}")
    print(f"  Polarité  : {du['polarite']}")
    if verbose:
        print(f"  Interp    : {du['interpretation'][:200]}")
        if du["interdictions"]:
            print(f"  Interdit  : {'; '.join(du['interdictions'][:3])}")
        if du["prescriptions"]:
            print(f"  À faire   : {'; '.join(du['prescriptions'][:3])}")
    print(sep)
