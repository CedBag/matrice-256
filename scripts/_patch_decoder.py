"""Patch TorchFaDecoder: inject forward_logits() method (second occurrence)."""
import pathlib

p = pathlib.Path(r"c:\Users\Cedoque BAGBONON\Projets\Matrice-256\bokono\ml\torch_layers.py")
data = p.read_bytes()

# Find the SECOND occurrence of "return self.activation(z)" (the Decoder one)
first = data.find(b"return self.activation(z)")
second = data.find(b"return self.activation(z)", first + 1)

if second == -1:
    print("ERREUR: deuxieme occurrence introuvable")
    raise SystemExit(1)

# The anchor starts at the second "return self.activation(z)" 
anchor = b"return self.activation(z)\r\n\r\n    @torch.no_grad()"
anchor_pos = data.find(anchor, second)

if anchor_pos == -1:
    print("ERREUR: anchor complet introuvable autour du Decoder")
    raise SystemExit(1)

replacement = (
    b"return self.activation(z)\r\n"
    b"\r\n"
    b"    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:\r\n"
    b'        """Retourne les logits bruts AVANT activation (TorchFaDual/Sigmoid).\r\n'
    b"        Usage exclusif : boucle d'entrainement avec BCEWithLogitsLoss.\r\n"
    b"        BCEWithLogitsLoss applique sigmoid en interne (log-sum-exp trick)\r\n"
    b'        pour une stabilite numerique maximale."""\r\n'
    b"        return torch.nn.functional.linear(x.to(torch.float32), self.weight, self.bias)\r\n"
    b"\r\n"
    b"    @torch.no_grad()"
)

new_data = data[:anchor_pos] + replacement + data[anchor_pos + len(anchor):]
p.write_bytes(new_data)
print("OK: forward_logits() injecte dans TorchFaDecoder (2e occurrence)")
