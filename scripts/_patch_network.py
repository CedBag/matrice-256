"""Patch TorchFaNet: inject forward_logits() and use it in fit() training loop."""
import pathlib

p = pathlib.Path(r"c:\Users\Cedoque BAGBONON\Projets\Matrice-256\bokono\ml\torch_network.py")
data = p.read_bytes()

# 1. Add forward_logits() method after forward() in TorchFaNet
anchor1 = b"            return self.decoder(h)\r\n\r\n    # "
anchor1_alt = b"            return self.decoder(h)\r\n\r\n    # \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 Entra"

# Find the right anchor
pos1 = data.find(anchor1_alt)
if pos1 == -1:
    pos1 = data.find(anchor1)
    
if pos1 == -1:
    print("ERREUR: anchor forward() introuvable")
    raise SystemExit(1)

# Find the end of "return self.decoder(h)" line
insert_pos = data.find(b"            return self.decoder(h)\r\n", pos1 - 50)
if insert_pos == -1:
    print("ERREUR: return self.decoder(h) introuvable")
    raise SystemExit(1)
insert_pos += len(b"            return self.decoder(h)\r\n")

new_method = (
    b"\r\n"
    b"    def forward_logits(self, x: torch.Tensor, return_dual: bool = False):\r\n"
    b'        """Retourne logits bruts (avant sigmoid) pour BCEWithLogitsLoss en training."""\r\n'
    b"        if return_dual:\r\n"
    b"            emb, attn, attn_gauche, attn_droit = self.embedding(x, return_dual=True)\r\n"
    b"            h = self.hidden(emb)\r\n"
    b"            return self.decoder.forward_logits(h), attn_gauche, attn_droit\r\n"
    b"        else:\r\n"
    b"            emb = self.embedding(x)\r\n"
    b"            h = self.hidden(emb)\r\n"
    b"            return self.decoder.forward_logits(h)\r\n"
)

data = data[:insert_pos] + new_method + data[insert_pos:]
print("OK: forward_logits() ajoute a TorchFaNet")

# 2. Replace self.forward(X_b, return_dual=True) with self.forward_logits(X_b, return_dual=True) in fit()
# There are 2 occurrences in the training loop (CPU path and CUDA path)
old_call = b"self.forward(X_b, return_dual=True)"
new_call = b"self.forward_logits(X_b, return_dual=True)"

count = data.count(old_call)
if count >= 2:
    data = data.replace(old_call, new_call)
    print(f"OK: {count} appels forward() remplaces par forward_logits() dans fit()")
else:
    print(f"ATTENTION: seulement {count} occurrence(s) trouvee(s) (attendu >= 2)")
    data = data.replace(old_call, new_call)

# 3. Also replace in validation section: self.forward(X_v, return_dual=True)
old_val = b"self.forward(X_v, return_dual=True)"
new_val = b"self.forward_logits(X_v, return_dual=True)"
val_count = data.count(old_val)
if val_count > 0:
    data = data.replace(old_val, new_val)
    print(f"OK: {val_count} appel(s) validation forward() -> forward_logits()")

p.write_bytes(data)
print("DONE: TorchFaNet.fit() utilise maintenant forward_logits() partout")
