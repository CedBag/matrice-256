"""
Script d'extraction complète — T3
Lit Les-256-Signe-de-Fa-Fon.pdf (source principale)
et extrait tout le texte page par page pour analyse.
"""
import fitz
import os
import json

DATA_DIR = r"c:\Users\Cedoque BAGBONON\Projets\Matrice-256\data"
PDF_MAIN = os.path.join(DATA_DIR, "Les-256-Signe-de-Fa-Fon.pdf")
PDF_OFFICIAL = os.path.join(DATA_DIR, "Les-256-Signes-Du-Fa-Officielpdf.pdf")
OUT_DIR = os.path.join(DATA_DIR, "extracted")
os.makedirs(OUT_DIR, exist_ok=True)

def extract_all_pages(filepath, out_filename):
    doc = fitz.open(filepath)
    pages_data = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        pages_data.append({
            "page": i + 1,
            "text": text,
            "char_count": len(text)
        })
    doc.close()
    out_path = os.path.join(OUT_DIR, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    return out_path, len(pages_data)

print("Extraction en cours...")

path1, n1 = extract_all_pages(PDF_MAIN, "fa_fon_raw.json")
print(f"✅ Les-256-Signe-de-Fa-Fon.pdf → {n1} pages extraites → {path1}")

path2, n2 = extract_all_pages(PDF_OFFICIAL, "fa_officiel_raw.json")
print(f"✅ Les-256-Signes-Du-Fa-Officielpdf.pdf → {n2} pages extraites → {path2}")

# Afficher les 5 premières entrées du fichier principal pour validation
with open(path1, encoding="utf-8") as f:
    data = json.load(f)

print("\n--- APERÇU : 5 premières pages avec contenu ---")
count = 0
for p in data:
    if p["char_count"] > 100:
        print(f"\n[Page {p['page']}] ({p['char_count']} chars)")
        print(p["text"][:600])
        print("...")
        count += 1
    if count >= 5:
        break
