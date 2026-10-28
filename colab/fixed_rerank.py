"""Fixed-weight final ordering (no learned ranker) -> submission.csv.
1. base order = records' fused score (fingerprint + contrastive + w_ap * analog, as written by infer_fp.py)
2. GLACIER / ICEBERG re-scoring inside same-formula groups of the top --top_n: members of one formula are re-ordered by
   z(base) + lam * z(ICE) + lam * z(GL) and keep the slots that formula already holds (the public 0.38 recipe)
3. library matches with cosine >= --lib_gate first
4. dedupe by the metric's identity (RDKit tautomer-canonical InChIKey-14), fill to 25
Validation records (with 'key') print MRR@25 before / after.
usage: python fixed_rerank.py --scores test_scores.pkl [--gl gl_test_out.pkl] --out submission.csv
"""
import argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument('--scores', required=True); p.add_argument('--gl', default=''); p.add_argument('--out', default='')
p.add_argument('--lam', type=float, default=1.0); p.add_argument('--top_n', type=int, default=40); p.add_argument('--lib_gate', type=float, default=0.8)
args = p.parse_args()
R = pd.read_pickle(args.scores); G = pd.read_pickle(args.gl) if args.gl else {}


def z(x):
    x = np.asarray(x, np.float64); f = np.isfinite(x)
    if f.sum() < 2: return np.zeros(len(x))
    m, s = x[f].mean(), x[f].std() + 1e-6
    return np.where(f, (x - m) / s, 0.0)


def order_of(r):
    keys = list(r.get('keys', []))
    if not keys: return [], {}
    smis = list(r['smiles']); forms = list(r['formula'])
    base = np.asarray(r['fused'], np.float64)
    order = list(np.argsort(-base, kind='stable')); info = dict(moved=0)
    sims = [G.get(n, {}).get(r['id'], {}) if isinstance(G, dict) else {} for n in ('ICE', 'GL')]
    if any(sims):
        top = order[:args.top_n]
        sv = [np.array([d.get(smis[i], np.nan) if d.get(smis[i]) is not None else np.nan for i in top], np.float64) for d in sims]
        zb = z(base[top]); comb = zb + sum(args.lam * z(v) for v in sv)
        new_top = list(top)
        for f in set(forms[i] for i in top):
            slots = [j for j, i in enumerate(top) if forms[i] == f]
            if len(slots) < 2: continue
            members = sorted(slots, key=lambda j: -comb[j])
            for s_, m_ in zip(slots, members): new_top[s_] = top[m_]
        info['moved'] = int(new_top[:25] != top[:25]); order = new_top + order[args.top_n:]
    ls = np.asarray(r['libsc'], np.float64)
    if args.lib_gate > 0 and (ls >= args.lib_gate).any():
        g = [i for i in order if ls[i] >= args.lib_gate]; g.sort(key=lambda i: -ls[i])
        order = g + [i for i in order if ls[i] < args.lib_gate]
    return order, info


from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); canon = rdMolStandardize.TautomerEnumerator()
def metric_key(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return None if m is None else Chem.MolToInchiKey(canon.Canonicalize(m))[:14]
    except Exception:
        return None


rows, rr0, rr1, moved, ndup = [], [], [], 0, 0
for _, r in R.iterrows():
    order, info = order_of(r); moved += info.get('moved', 0)
    smis = list(r.get('smiles', [])); keys = list(r.get('keys', []))
    seen, out, okeys = set(), [], []
    for i in order:
        k = metric_key(smis[i]) or keys[i]
        if k in seen: ndup += 1; continue
        seen.add(k); out.append(smis[i]); okeys.append(keys[i])
        if len(out) == 25: break
    rows.append(dict(molecule_id=r['id'], smiles=';'.join(out) if out else 'C'))
    if r.get('key') is not None:
        t = r['key']; base = np.asarray(r['fused']) if keys else np.zeros(0); bo = [keys[i] for i in np.argsort(-base, kind='stable')][:25]
        rr0.append(1 / (bo.index(t) + 1) if t in bo else 0.0); rr1.append(1 / (okeys.index(t) + 1) if t in okeys else 0.0)
print('molecules', len(rows), '| simulator re-order changed the top 25 of', moved, '| tautomer duplicates skipped', ndup)
if rr0:
    S = pd.DataFrame(dict(set=R['set'].values[:len(rr0)], before=rr0, after=rr1)); print(S.groupby('set').mean().round(4))
if args.out:
    pd.DataFrame(rows).to_csv(args.out, index=False); print('written', args.out)
print('RERANK DONE')
