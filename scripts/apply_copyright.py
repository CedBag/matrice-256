import os

header = """# Copyright (c) 2026 Cedoque BAGBONON — Projet Matrice-256.
# Tous droits réservés. Reproduction ou distribution interdite sans autorisation.

"""

for d in ['bokono', 'tests']:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'Copyright (c) 2026 Cedoque BAGBONON' not in content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(header + content)
                    print(f'Copyright ajouté : {path}')
