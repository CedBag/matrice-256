"""
Script d'audit des PDFs Fâ — T1
Lit les 3 PDFs et produit un rapport de structure/contenu.
"""
import fitz  # PyMuPDF
import os
import sys

DATA_DIR = r"c:\Users\Cedoque BAGBONON\Projets\Matrice-256\data"
PDFS = [
    "Fâ.pdf",
    "Les-256-Signe-de-Fa-Fon.pdf",
    "Les-256-Signes-Du-Fa-Officielpdf.pdf"
]

def audit_pdf(filepath):
    doc = fitz.open(filepath)
    total_pages = len(doc)
    sample_pages = []
    for i in [0, 1, 2, min(10, total_pages-1), min(20, total_pages-1)]:
        if i < total_pages:
            page = doc[i]
            text = page.get_text().strip()
            sample_pages.append({
                "page": i + 1,
                "chars": len(text),
                "preview": text[:400].replace('\n', ' | ')
            })
    doc.close()
    return {"total_pages": total_pages, "samples": sample_pages}

print("=" * 70)
print("AUDIT DES PDFs Fâ — Projet bokono / Matrice-256")
print("=" * 70)

for pdf_name in PDFS:
    path = os.path.join(DATA_DIR, pdf_name)
    print(f"\n{'─'*70}")
    print(f"📄 FICHIER : {pdf_name}")
    print(f"   Taille  : {os.path.getsize(path) / 1024:.1f} Ko")

    try:
        info = audit_pdf(path)
        print(f"   Pages   : {info['total_pages']}")
        for s in info['samples']:
            print(f"\n   [Page {s['page']}] ({s['chars']} caractères)")
            print(f"   {s['preview'][:300]}")
    except Exception as e:
        print(f"   ERREUR : {e}")

print(f"\n{'='*70}")
print("Audit terminé.")
