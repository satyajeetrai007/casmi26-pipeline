"""Natural-product-likeness of validation structures (CPU): for every validation key, the max Tanimoto to any COCONUT
structure (Morgan r2 4096 bits folded to 1024), its element set and PubChem presence. Output /kaggle/working/np_like.csv.
Used to build test-like class-2 subsets: no public spectra (S2) AND natural-product-like.
"""
import numpy as np, pandas as pd, torch, time
from fpmodel import formula_elements, ELEMENTS
t0 = time.time()
Q = pd.read_pickle('/content/work/val_queries_seed0.pkl')
pool = pd.read_parquet('/content/work/pools/pool.parquet', columns=['ik14', 'formula', 'src_mask', 'mass'])
bits = np.load('/content/work/pools/fp_morgan4096.npy', mmap_mode='r')
row = pd.Series(np.arange(len(pool)), index=pool.ik14.values); row = row[~row.index.duplicated()]
coco = np.where((pool.src_mask.values & 2) > 0)[0]
def folded(rows):
    b = np.unpackbits(np.asarray(bits[np.sort(rows)]), axis=1)[:, :4096].reshape(len(rows), 4, 1024).max(1)
    return torch.from_numpy(b.astype(np.float32)), np.sort(rows)
C, _ = folded(coco); Cn = C.sum(1)
keys = Q.key.unique(); qrows = np.array([row.get(k, -1) for k in keys]); ok = qrows >= 0
X, xr = folded(qrows[ok]); Xn = X.sum(1); best = torch.zeros(len(xr))
for c0 in range(0, len(C), 50000):
    inter = X @ C[c0:c0 + 50000].T; tan = inter / (Xn[:, None] + Cn[None, c0:c0 + 50000] - inter).clamp(min=1)
    best = torch.maximum(best, tan.max(1).values)
tan_of = dict(zip(pool.ik14.values[xr], best.numpy()))
out = []
for k in keys:
    r = row.get(k, -1); f = pool.formula.values[r] if r >= 0 else ''
    el = formula_elements(f); hal = el[ELEMENTS.index('F')] + el[ELEMENTS.index('Cl')] + el[ELEMENTS.index('Br')] + el[ELEMENTS.index('I')]
    out.append(dict(key=k, sets='/'.join(sorted(Q.loc[Q.key == k, 'set'].unique())), coco_tanimoto=float(tan_of.get(k, np.nan)), halogens=int(hal),
                    n_O=int(el[ELEMENTS.index('O')]), n_N=int(el[ELEMENTS.index('N')]), mass=float(pool.mass.values[r]) if r >= 0 else np.nan))
D = pd.DataFrame(out); D.to_csv('/kaggle/working/np_like.csv', index=False)
for s, g in D.groupby('sets'):
    print(s, len(g), 'coco_tanimoto quantiles', g.coco_tanimoto.quantile([.25, .5, .75, .9]).round(3).tolist(), '| halogen-free', round((g.halogens == 0).mean(), 3))
s2 = D[D.sets == 'S2']
for th in (0.5, 0.6, 0.7):
    print(f'S2 NP-like (no halogens, coco_tanimoto >= {th}):', int(((s2.halogens == 0) & (s2.coco_tanimoto >= th)).sum()))
print('NP SUBSET DONE', f'{time.time()-t0:.0f}s')
