"""Candidate-structure features for the cross-encoder (own code). One function, used identically for training
(pool structures) and inference (pool + PubChem candidates), so every candidate goes through the same standardisation.

featurize(smiles) -> dict(atoms [A,5] int8, frags [F,2] float32, cnt [2048] uint8, n_atoms, n_frags, mass) or None
  atoms: element index (1..12, 13 = other), heavy degree, total H, aromatic, in-ring
  frags: neutral masses of the two pieces of each 1-bond (acyclic) or 2-bond (same ring) cleavage, larger first
  cnt:   Morgan r2 count fingerprint (2048, clipped 255) of the standardised molecule
Standardisation: largest organic fragment + uncharge (tautomers left as written).
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
A_MAX, F_MAX = 64, 64
ELEM = {'C': 1, 'N': 2, 'O': 3, 'S': 4, 'P': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'Si': 10, 'B': 11, 'Se': 12}
MONO = {'C': 12.0, 'N': 14.003074, 'O': 15.994915, 'S': 31.972071, 'P': 30.973762, 'F': 18.998403, 'Cl': 34.968853, 'Br': 78.918337,
        'I': 126.904473, 'Si': 27.976927, 'B': 11.009305, 'Se': 79.916522, 'Na': 22.989769, 'K': 38.963707}
H = 1.007825
_LFC = rdMolStandardize.LargestFragmentChooser(); _UNCH = rdMolStandardize.Uncharger()
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _components_mass(n, adj, amass, removed):
    seen = [-1] * n; masses = []
    for s in range(n):
        if seen[s] >= 0: continue
        cid = len(masses); stack = [s]; seen[s] = cid; m = 0.0
        while stack:
            u = stack.pop(); m += amass[u]
            for v, bid in adj[u]:
                if bid in removed or seen[v] >= 0: continue
                seen[v] = cid; stack.append(v)
        masses.append(m)
    return masses


def featurize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = _UNCH.uncharge(_LFC.choose(mol))
        n = mol.GetNumAtoms()
        if n == 0:
            return None
        atoms = np.zeros((A_MAX, 5), np.int8); amass = []
        for a in mol.GetAtoms():
            sym = a.GetSymbol(); hs = a.GetTotalNumHs(); amass.append(MONO.get(sym, a.GetMass()) + hs * H)
            i = a.GetIdx()
            if i < A_MAX:
                atoms[i] = (ELEM.get(sym, 13), min(a.GetDegree(), 7), min(hs, 5), int(a.GetIsAromatic()), int(a.IsInRing()))
        M = float(sum(amass))
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))
        frags = []; seen = set()
        def add(ms):
            if len(ms) != 2: return
            a, b = max(ms), min(ms); k = round(a, 3)
            if k in seen: return
            seen.add(k); frags.append((a, b))
        for b in mol.GetBonds():
            if not b.IsInRing():
                add(_components_mass(n, adj, amass, {b.GetIdx()}))
        if len(frags) < F_MAX:
            for ring in mol.GetRingInfo().BondRings():
                r = list(ring)
                for x in range(len(r)):
                    for y in range(x + 2, len(r)):          # non-adjacent bond pairs open the ring into two pieces
                        if len(frags) >= 2 * F_MAX: break
                        add(_components_mass(n, adj, amass, {r[x], r[y]}))
        frags.sort(key=lambda t: -t[1])                      # most balanced cleavages first
        fr = np.zeros((F_MAX, 2), np.float32); nf = min(len(frags), F_MAX)
        if nf: fr[:nf] = np.array(frags[:nf], np.float32)
        cnt = np.clip(_GEN.GetCountFingerprintAsNumPy(mol), 0, 255).astype(np.uint8)
        return dict(atoms=atoms, frags=fr, cnt=cnt, n_atoms=min(n, A_MAX), n_frags=nf, mass=M)
    except Exception:
        return None


def featurize_many(smiles_list):
    """-> stacked arrays (atoms [N,A,5], frags [N,F,2], cnt [N,2048], n [N,2] int16, ok [N] bool)."""
    N = len(smiles_list)
    atoms = np.zeros((N, A_MAX, 5), np.int8); frags = np.zeros((N, F_MAX, 2), np.float32); cnt = np.zeros((N, 2048), np.uint8)
    nn = np.zeros((N, 2), np.int16); ok = np.zeros(N, bool)
    for i, s in enumerate(smiles_list):
        f = featurize(s)
        if f is None: continue
        atoms[i] = f['atoms']; frags[i] = f['frags']; cnt[i] = f['cnt']; nn[i] = (f['n_atoms'], f['n_frags']); ok[i] = True
    return atoms, frags, cnt, nn, ok
