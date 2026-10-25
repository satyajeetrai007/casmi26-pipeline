"""MetFrag-style in-silico fragmentation scorer (own implementation).
For a candidate SMILES: enumerate fragments obtained by cleaving 1 or 2 bonds (acyclic single bonds; ring bonds
only in pairs within the same ring), compute neutral fragment masses, and explain the measured peaks
(fragment mass + adduct shift + h*1.007825 for h in -2..+3, within tol). Score = weighted explained intensity with a
parsimony weight (1-cleavage fragments weigh 1.0, 2-cleavage 0.6) ; also returns the fraction of peaks explained.
Forum ablations (NeckBeard, CASMI26 discussion 743254): 2-bond cuts x0.6, linear intensity, 0.005 Da, H -2..+3.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
H_MASS = 1.007825; PROTON = 1.007276
ADDUCT_ION_SHIFT = {'[M+H]+': PROTON, '[M+NH4]+': 18.033823, '[M-H2O+H]+': PROTON - 18.010565, '[M-2H2O+H]+': PROTON - 2 * 18.010565,
                    '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -PROTON, '[M-H2O-H]-': -PROTON - 18.010565,
                    '[M+CH2O2-H]-': 46.005479 - PROTON, '[M+Cl]-': 34.969402}
FRAG_SHIFT = {'positive': PROTON, 'negative': -PROTON}      # fragments observed as [F+H]+ / [F-H]-
_cache = {}


def fragment_masses(smiles, max_atoms=120):
    """-> (masses float64 array, n_cleavages int8 array) of neutral fragments (incl. the intact molecule, 0 cleavages)."""
    if smiles in _cache:
        return _cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        _cache[smiles] = (np.zeros(0), np.zeros(0, np.int8)); return _cache[smiles]
    mol = Chem.AddHs(mol)
    heavy = np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()])
    amass = np.array([a.GetMass() for a in mol.GetAtoms()])          # average masses -> replace by mono
    mono = {1: 1.007825, 6: 12.0, 7: 14.003074, 8: 15.994915, 9: 18.998403, 15: 30.973762, 16: 31.972071, 17: 34.968853, 35: 78.918337, 53: 126.904473, 14: 27.976927, 5: 11.009305, 34: 79.916522, 11: 22.989769, 19: 38.963707}
    amass = np.array([mono.get(a.GetAtomicNum(), a.GetMass()) for a in mol.GetAtoms()])
    n = mol.GetNumAtoms()
    bonds = [(b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.IsInRing()) for b in mol.GetBonds()
             if heavy[b.GetBeginAtomIdx()] and heavy[b.GetEndAtomIdx()]]
    ri = mol.GetRingInfo(); bond_rings = ri.BondRings()
    ring_of = {}
    for rid, br in enumerate(bond_rings):
        for b in br:
            ring_of.setdefault(b, set()).add(rid)
    # adjacency
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))

    def components(removed):
        seen = np.zeros(n, bool); comps = []
        for s in range(n):
            if seen[s]: continue
            stack = [s]; seen[s] = True; comp = []
            while stack:
                u = stack.pop(); comp.append(u)
                for v, bid in adj[u]:
                    if bid in removed or seen[v]: continue
                    seen[v] = True; stack.append(v)
            comps.append(comp)
        return comps
    masses = [float(amass.sum())]; ncl = [0]; seen_masses = {round(masses[0], 4)}
    acyclic = [b for b in bonds if not b[3]]; ring = [b for b in bonds if b[3]]
    # single acyclic cleavages
    singles = []
    for bid, i, j, _ in acyclic:
        comps = components({bid})
        if len(comps) == 2:
            singles.append(bid)
            for comp in comps:
                m = float(amass[comp].sum()); k = round(m, 4)
                if k not in seen_masses and len([c for c in comp if heavy[c]]) >= 2:
                    seen_masses.add(k); masses.append(m); ncl.append(1)
    # double cleavages: two acyclic bonds, or two bonds of the same ring (ring opening), or acyclic + ring pair capped
    pairs = []
    for a in range(len(singles)):
        for b in range(a + 1, len(singles)):
            pairs.append((singles[a], singles[b]))
    for rid, br in enumerate(bond_rings):
        br = [b for b in br if b in {x[0] for x in ring}]
        for a in range(len(br)):
            for b in range(a + 1, len(br)):
                pairs.append((br[a], br[b]))
    if len(pairs) > 4000:
        pairs = pairs[:4000]
    for p in pairs:
        comps = components(set(p))
        if len(comps) < 2: continue
        for comp in comps:
            if len([c for c in comp if heavy[c]]) < 2: continue
            m = float(amass[comp].sum()); k = round(m, 4)
            if k not in seen_masses:
                seen_masses.add(k); masses.append(m); ncl.append(2)
    out = (np.array(masses), np.array(ncl, np.int8)); _cache[smiles] = out
    return out


def frag_score(smiles, spectra, tol=0.005, w2=0.6, hshift=(-2, -1, 0, 1, 2, 3)):
    """spectra: list of dict(mz, it, adduct, mode). Returns (explained_intensity_weighted, explained_intensity, frac_peaks_explained, n_frag)."""
    masses, ncl = fragment_masses(smiles)
    if len(masses) == 0:
        return 0.0, 0.0, 0.0, 0
    w_frag = np.where(ncl == 0, 1.0, np.where(ncl == 1, 1.0, w2))
    tot_w, tot_e, tot_p, cnt = 0.0, 0.0, 0.0, 0
    for s in spectra:
        mz = np.asarray(s['mz'], float); it = np.asarray(s['it'], float)
        if len(mz) == 0: continue
        shift = FRAG_SHIFT['positive' if s['mode'] == 'positive' else 'negative']
        # candidate ion masses for all fragments and H shifts
        ion = (masses[:, None] + shift + H_MASS * np.array(hshift)[None, :]).ravel(); wf = np.repeat(w_frag, len(hshift))
        o = np.argsort(ion); ion = ion[o]; wf = wf[o]
        lo = np.searchsorted(ion, mz - tol); hi = np.searchsorted(ion, mz + tol, side='right')
        best = np.zeros(len(mz))
        for k in range(len(mz)):
            if hi[k] > lo[k]: best[k] = wf[lo[k]:hi[k]].max()
        tot_w += float((it * best).sum() / it.sum()); tot_e += float((it * (best > 0)).sum() / it.sum()); tot_p += float((best > 0).mean()); cnt += 1
    if cnt == 0:
        return 0.0, 0.0, 0.0, len(masses)
    return tot_w / cnt, tot_e / cnt, tot_p / cnt, len(masses)
