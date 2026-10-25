"""Join PubChem popularity counts (dmitriigluzdov/casmi26-pubchem-popularity-prior) onto the candidate pool, and
collect per-validation-query PubChem candidates (10 ppm window, not in the pool) for the PubChem-channel experiment.
Outputs: /content/work/pop/pool_pop.parquet (row-aligned with pools/pool.parquet), /content/work/pop/val_pubchem.pkl
"""
import os, time, glob, numpy as np, pandas as pd
t0 = time.time()
POP = sorted(glob.glob('/kaggle/input/**/pc_ik14.npy', recursive=True), key=len)[0]; POP = os.path.dirname(POP)
TIER = os.path.dirname(sorted(glob.glob('/kaggle/input/**/casmi26-pubchem-tier/pc_smiles.npy', recursive=True), key=len)[0])
OUT = '/content/work/pop'; os.makedirs(OUT, exist_ok=True)
ik = np.load(f'{POP}/pc_ik14.npy', mmap_mode='r'); print('pc_ik14', ik.shape, ik.dtype, ik[:2], flush=True)
N = ik.shape[0]
def col(name, dtype=None):
    p = f'{POP}/{name}'
    return np.load(p, mmap_mode='r') if os.path.exists(p) else None
lsid, lpmid, lpat, npf, nv = col('pc_lsid.npy'), col('pc_lpmid.npy'), col('pc_lpatent.npy'), col('pc_np.npy'), col('pc_n_vendors.npy')
extra = {n: col(f'pc_{n}.npy') for n in ('lotus_refs', 'lmesh', 'lsynonyms', 'n_depositors', 'src_mask', 'formula_isomers')}
for n, a in [('lsid', lsid), ('lpmid', lpmid), ('lpat', lpat), ('np', npf), ('nv', nv)] + list(extra.items()):
    print(n, None if a is None else (a.shape, a.dtype, np.asarray(a[:3])), flush=True)
# keys as fixed-width bytes (aligned with pc_mass order: no global sort needed)
keys = np.asarray(ik).astype('S14') if ik.dtype.kind in 'SU' else np.ascontiguousarray(ik).view('S14').reshape(-1)
pc_mass = np.load(f'{TIER}/pc_mass.npy', mmap_mode='r'); print('keys', keys.shape, keys[:2], 'pc_mass', pc_mass.shape, f'{time.time()-t0:.0f}s', flush=True)
pool = pd.read_parquet('/content/work/pools/pool.parquet', columns=['ik14', 'mass'])
q = pool.ik14.values.astype('S14'); pm = pool.mass.values
lo = np.searchsorted(pc_mass, pm * (1 - 3e-6)); hi = np.searchsorted(pc_mass, pm * (1 + 3e-6), side='right')
hit = np.zeros(len(pool), bool); src = np.zeros(len(pool), np.int64)
for i in range(len(pool)):
    a, b = lo[i], hi[i]
    if b > a:
        w = keys[a:b]; m = np.where(w == q[i])[0]
        if len(m):
            hit[i] = True; src[i] = a + m[0]
    if i % 200000 == 0:
        print(f'  join {i}/{len(pool)} {time.time()-t0:.0f}s', flush=True)
print('pool rows', len(pool), 'found in PubChem', int(hit.sum()), f'{time.time()-t0:.0f}s', flush=True)
def take(a, default=0.0):
    if a is None:
        return np.full(len(pool), default, np.float32)
    v = np.asarray(a[src]).astype(np.float32); v[~hit] = default; return v
pp = pd.DataFrame({'ik14': pool.ik14.values, 'in_pubchem': hit, 'lsid': take(lsid), 'lpmid': take(lpmid), 'lpatent': take(lpat), 'np_flag': take(npf), 'n_vendors': take(nv)})
for n, a in extra.items():
    pp[n] = take(a)
pp.to_parquet(f'{OUT}/pool_pop.parquet'); print(pp.describe().T, flush=True)
# per-validation-query PubChem windows (not in pool), capped by popularity
Q = pd.read_pickle('/content/work/val_queries_seed0.pkl'); meta = pd.read_parquet('/content/work/cache/spec_meta.parquet', columns=['precursor_mz', 'adduct'])
from casmi_val import neutral_mass
nm = neutral_mass(meta.precursor_mz.values, meta.adduct.values)
pc_off = np.load(f'{TIER}/pc_off.npy', mmap_mode='r'); pc_smi = np.load(f'{TIER}/pc_smiles.npy', mmap_mode='r')
pool_keys = set(pool.ik14.values); out = {}; CAP = 1500
for qi, r in Q.iterrows():
    target = float(np.nanmedian(nm[r['idx']])); lo, hi = np.searchsorted(pc_mass, [target * (1 - 10e-6), target * (1 + 10e-6)])
    cand = np.arange(lo, hi)
    if len(cand) == 0:
        out[f"{r['set']}|{r['key']}"] = None; continue
    k = keys[cand].astype(str); keep = np.array([x not in pool_keys for x in k])
    cand, k = cand[keep], k[keep]
    popv = (np.asarray(lsid[cand]).astype(np.float32) if lsid is not None else np.zeros(len(cand))) + (np.asarray(lpmid[cand]).astype(np.float32) if lpmid is not None else 0)
    sel = np.argsort(-popv)[:CAP]; cand, k, popv = cand[sel], k[sel], popv[sel]
    smi = [bytes(pc_smi[pc_off[i]:pc_off[i + 1]]).decode() for i in cand]
    out[f"{r['set']}|{r['key']}"] = dict(ik14=k, smiles=smi, pop=popv, mass=np.asarray(pc_mass[cand]), n_window=int(hi - lo), truth_in_window=bool((keys[lo:hi].astype(str) == r['key']).any()))
    if qi % 200 == 0:
        print(f'  {qi}/{len(Q)} {time.time()-t0:.0f}s', flush=True)
pd.to_pickle(out, f'{OUT}/val_pubchem.pkl')
tw = [v['truth_in_window'] for v in out.values() if v]; print('truth found in PubChem window (any):', np.mean(tw), 'n', len(tw))
print('POP DONE', f'{time.time()-t0:.0f}s')
