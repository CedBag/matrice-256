"""
Parser structuré des 256 du — T3 (suite)
Analyse fa_fon_raw.json et construit fa_knowledge.json
conforme au schéma défini en T2.
"""
import json
import re
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

DATA_DIR = r"c:\Users\Cedoque BAGBONON\Projets\Matrice-256\data"
EXTRACTED_DIR = os.path.join(DATA_DIR, "extracted")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)

# Import du référentiel canonique — source unique de vérité
from bokono.core.fa_encoder import KPOLI_16 as _KPOLI_DICTS
KPOLI_16 = [k["nom"] for k in _KPOLI_DICTS]  # ["GBE", "YEKU", ..., "IKA"]

def kpoli_index(name):
    """Retourne l'index (0-15) du kpoli par nom."""
    for i, k in enumerate(KPOLI_16):
        if name.upper().startswith(k):
            return i
    return -1

def encode_binary(left_idx, right_idx):
    """Encode deux kpoli en 8 bits."""
    return f"{left_idx:04b}{right_idx:04b}"

def infer_polarity(text):
    """Infère la polarité du signe depuis son texte."""
    text_lower = text.lower()
    pos_words = ["protégé", "bonheur", "richesse", "succès", "bénédiction",
                 "chance", "prospérité", "santé", "victoire", "joie", "bien"]
    neg_words = ["danger", "mort", "malheur", "problème", "sorcellerie",
                 "ennemi", "maladie", "pauvreté", "malédiction", "échec"]
    pos = sum(1 for w in pos_words if w in text_lower)
    neg = sum(1 for w in neg_words if w in text_lower)
    if pos > neg + 1:
        return "favorable"
    elif neg > pos + 1:
        return "défavorable"
    return "neutre"

def extract_interdictions(text):
    """Extrait les interdictions du texte d'un du."""
    interdictions = []
    patterns = [
        r"ne doit pas (.+?)(?:\.|;|\n)",
        r"il est interdit de (.+?)(?:\.|;|\n)",
        r"doit éviter (.+?)(?:\.|;|\n)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        interdictions.extend([m.strip() for m in matches if len(m.strip()) > 5])
    return interdictions[:5]  # Max 5 interdictions

def extract_prescriptions(text):
    """Extrait les prescriptions (ce qu'il faut faire)."""
    prescriptions = []
    patterns = [
        r"doit faire (.+?)(?:\.|;|\n)",
        r"il faut (.+?)(?:\.|;|\n)",
        r"doit (.+?)(?:sacrifice|rituel|offrande)(.+?)(?:\.|;|\n)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        prescriptions.extend([m[0].strip() if isinstance(m, tuple) else m.strip()
                               for m in matches if len(str(m).strip()) > 5])
    return prescriptions[:5]

def parse_du_from_pages(pages):
    """
    Parse les 256 du depuis les pages extraites.
    Stratégie : détecte les patterns "N-NOM MEDJI/..." qui marquent le début d'un du.
    """
    all_du = {}
    current_id = None
    current_name = None
    current_text_parts = []

    # Pattern pour détecter le début d'un signe
    # Ex: "1-GBE MEDJI", "17- GBE YEKU", "256-IKA OTURA"
    sign_header = re.compile(
        r"^(\d{1,3})[–\-]\s*([A-ZÀÂÉÈÊËÎÏÔÙÛÜ][A-ZÀÂÉÈÊËÎÏÔÙÛÜ\s\-\']{2,40})",
        re.MULTILINE
    )

    full_text = "\n".join(p["text"] for p in pages if p.get("char_count", 0) > 50)

    matches = list(sign_header.finditer(full_text))

    for i, match in enumerate(matches):
        sign_id = int(match.group(1))
        sign_name = match.group(2).strip()

        # Récupère le texte jusqu'au prochain signe
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()

        if 1 <= sign_id <= 256:
            all_du[sign_id] = {
                "id": sign_id - 1,           # 0-indexé (0 à 255)
                "numero": sign_id,            # 1-indexé (1 à 256)
                "nom_fon": sign_name,
                "texte_brut": body[:2000],    # Limité pour JSON propre
                "polarite": infer_polarity(body),
                "interdictions": extract_interdictions(body),
                "prescriptions": extract_prescriptions(body),
                "interpretation": body[:500].replace('\n', ' ').strip(),
            }

    return all_du

# --- Chargement des données ---
with open(os.path.join(EXTRACTED_DIR, "fa_fon_raw.json"), encoding="utf-8") as f:
    pages_fon = json.load(f)

print(f"Pages source chargées : {len(pages_fon)}")

# --- Parsing ---
du_dict = parse_du_from_pages(pages_fon)
print(f"Du détectés : {len(du_dict)}")

# --- Construction de la base de connaissance complète (256 entrées) ---
knowledge_base = []

# Ordre canonique des 256 du : GBE×GBE, GBE×YEKU, ..., IKA×IKA
du_id_counter = 0
for left_idx in range(16):
    for right_idx in range(16):
        numero = du_id_counter + 1
        left_name = KPOLI_16[left_idx]
        right_name = KPOLI_16[right_idx]

        # Nom canonique
        if left_idx == right_idx:
            nom_canonique = f"{left_name}-MEDJI"
        else:
            nom_canonique = f"{left_name}-{right_name}"

        bin_code = encode_binary(left_idx, right_idx)

        # Fusion avec les données extraites du PDF si disponibles
        extracted = du_dict.get(numero, {})

        entry = {
            "id": du_id_counter,
            "numero": numero,
            "nom_canonique": nom_canonique,
            "nom_fon": extracted.get("nom_fon", nom_canonique),
            "code_binaire": bin_code,
            "valeur_decimale": du_id_counter,
            "kpoli_gauche": left_name,
            "kpoli_droit": right_name,
            "index_gauche": left_idx,
            "index_droit": right_idx,
            "est_meji": left_idx == right_idx,  # Signe double (le plus puissant)
            "polarite": extracted.get("polarite", "non_défini"),
            "intensite": round(0.5 + (0.5 * (left_idx + right_idx) / 30), 2),
            "domaines": [],
            "interpretation": extracted.get("interpretation", ""),
            "prescriptions": extracted.get("prescriptions", []),
            "interdictions": extracted.get("interdictions", []),
            "proverbes": [],
            "texte_source": extracted.get("texte_brut", ""),
            "correspondance_medicale": None,
            "correspondance_agricole": None,
            "vecteur_semantique": None,  # Semaine 2 : embedding
        }
        knowledge_base.append(entry)
        du_id_counter += 1

# --- Sauvegarde ---
out_path = os.path.join(KB_DIR, "fa_knowledge.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

# --- Rapport de qualité ---
total = len(knowledge_base)
with_interp = sum(1 for d in knowledge_base if len(d["interpretation"]) > 20)
with_interdictions = sum(1 for d in knowledge_base if len(d["interdictions"]) > 0)
meji_count = sum(1 for d in knowledge_base if d["est_meji"])
favorable = sum(1 for d in knowledge_base if d["polarite"] == "favorable")
defavorable = sum(1 for d in knowledge_base if d["polarite"] == "défavorable")
neutre = sum(1 for d in knowledge_base if d["polarite"] == "neutre")
non_defini = sum(1 for d in knowledge_base if d["polarite"] == "non_défini")

print(f"\n{'='*60}")
print(f"BASE DE CONNAISSANCE fa_knowledge.json — RAPPORT")
print(f"{'='*60}")
print(f"Du total générés    : {total}/256")
print(f"Avec interprétation : {with_interp} ({with_interp/total*100:.0f}%)")
print(f"Avec interdictions  : {with_interdictions} ({with_interdictions/total*100:.0f}%)")
print(f"Signes Meji (doubles): {meji_count} (attendu: 16)")
print(f"Polarité favorable  : {favorable}")
print(f"Polarité défavorable: {defavorable}")
print(f"Polarité neutre     : {neutre}")
print(f"Non défini          : {non_defini}")
print(f"\nFichier sauvegardé  : {out_path}")
print(f"{'='*60}")

# Aperçu de 3 entrées
print("\n--- APERÇU : 3 premiers du ---")
for d in knowledge_base[:3]:
    print(f"\n[{d['numero']}] {d['nom_fon']} | Binaire: {d['code_binaire']} | Polarité: {d['polarite']}")
    print(f"    Interp: {d['interpretation'][:150]}...")
    print(f"    Interdictions: {d['interdictions'][:2]}")
