import json
import os
import shutil

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'knowledge_base')
ORIGINAL_KB = os.path.join(DATA_DIR, 'fa_knowledge.json')
PRIVATE_KB = os.path.join(DATA_DIR, 'fa_knowledge_private.json')
PUBLIC_KB = os.path.join(DATA_DIR, 'fa_knowledge_public.json')

def run_security_split():
    if not os.path.exists(ORIGINAL_KB) and not os.path.exists(PRIVATE_KB):
        print("❌ Aucune base de données trouvée à scinder.")
        return

    # Si le script a déjà tourné, on part du fichier privé
    source_file = PRIVATE_KB if os.path.exists(PRIVATE_KB) else ORIGINAL_KB
    
    with open(source_file, 'r', encoding='utf-8') as f:
        full_kb = json.load(f)
    
    # 1. Sauvegarder/Renommer le fichier ORIGINAL vers PRIVATE s'il existe
    if os.path.exists(ORIGINAL_KB) and not os.path.exists(PRIVATE_KB):
        shutil.copy2(ORIGINAL_KB, PRIVATE_KB)
        print(f"✅ Base de connaissances privée sécurisée : {PRIVATE_KB}")
    
    # 2. Créer l'objet "Public" en filtrant chirurgicalement (aucune interprétation, pôle neutre)
    public_kb = []
    
    public_allowed_keys = [
        "id", "numero", "nom_canonique", "nom_fon", "code_binaire",
        "valeur_decimale", "kpoli_gauche", "kpoli_droit",
        "index_gauche", "index_droit", "est_meji"
    ]
    
    for du in full_kb:
        safe_du = {k: du[k] for k in public_allowed_keys if k in du}
        
        # Override de sécurité pour FaNet open-source (fallback dummy values)
        safe_du["polarite"] = "neutre"
        safe_du["intensite"] = 0.5
        
        # Supprime formellement tout texte d'interprétation pour éviter les bots d'IA
        # On ne transmet RIEN du texte divinatoire
        
        public_kb.append(safe_du)

    # 3. Ecrire le fichier public
    with open(PUBLIC_KB, 'w', encoding='utf-8') as f:
        json.dump(public_kb, f, ensure_ascii=False, indent=2)
    print(f"✅ Base de connaissances publique expurgée créée : {PUBLIC_KB}")
    
    # 4. Supprimer le fichier original non-sécurisé
    if os.path.exists(ORIGINAL_KB):
        os.remove(ORIGINAL_KB)
        print(f"✅ Fichier original fa_knowledge.json (non-GIT-ignored) a été détruit pour protéger le secret.")

if __name__ == "__main__":
    print("--- DEBUT PROCEDURE SPLIT KB ---")
    run_security_split()
    print("--- PROCEDURE TERMINEE ---")
