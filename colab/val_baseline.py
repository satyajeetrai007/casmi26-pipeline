"""Baseline numbers for the validation sets: cosine library search (class 1) and candidate-pool coverage (class 2)."""
import time, numpy as np, pandas as pd, pickle
from casmi_val import *

t0 = time.time()
meta = load_meta()
pool = pd.read_parquet('/content/work/pools/pool.parquet')
coco_keys = pool.loc[(pool.src_mask & 2) > 0, 'ik14'].values
Q = build_sets(meta, seed=0, coconut_keys=coco_keys)
Q.to_pickle('/content/work/val_queries_seed0.pkl')
print(Q.groupby('set').size(), 'queries spectra', Q.idx.map(len).describe()[['mean', '50%', 'max']].to_dict(), f'{time.time()-t0:.0f}s', flush=True)

MZ = np.load(f'{CACHE}/peaks_mz.npy', mmap_mode='r'); IT = np.load(f'{CACHE}/peaks_int.npy', mmap_mode='r'); NPK = np.load(f'{CACHE}/n_peaks.npy')
nm_all = neutral_mass(meta.precursor_mz.values, meta.adduct.values)
lib_ok = meta.adduct.isin(TEST_ADDUCTS).values & np.isfinite(nm_all) & (NPK >= 3)
lib_idx = np.where(lib_ok)[0]; lib_nm = nm_all[lib_idx]; order = np.argsort(lib_nm); lib_idx = lib_idx[order]; lib_nm = lib_nm[order]
lib_key = meta.inchikey14.values[lib_idx]; lib_lib = meta.ingest_lib.values[lib_idx]; lib_mode = meta.ionization_mode.values[lib_idx]
print('library spectra', len(lib_idx), f'{time.time()-t0:.0f}s', flush=True)


def spec(i):
    n = int(NPK[i]); return np.asarray(MZ[i, :n], np.float64), np.sqrt(np.asarray(IT[i, :n], np.float64))


def cosine(qmz, qit, cmz, cit, tol=0.02):
    if len(cmz) == 0 or len(qmz) == 0:
        return 0.0
    lo = np.searchsorted(cmz, qmz - tol); hi = np.searchsorted(cmz, qmz + tol, side='right')
    pairs = [(qit[i] * cit[j], i, j) for i in range(len(qmz)) for j in range(lo[i], hi[i])]
    if not pairs:
        return 0.0
    pairs.sort(reverse=True); used_q, used_c, s = set(), set(), 0.0
    for v, i, j in pairs:
        if i in used_q or j in used_c: continue
        used_q.add(i); used_c.add(j); s += v
    return float(s / (np.linalg.norm(qit) * np.linalg.norm(cit) + 1e-12))


def search(q, mode, ppm=10.0, topk=25):
    idx = q['idx']; nms = nm_all[idx]; target = float(np.median(nms[np.isfinite(nms)]))
    lo, hi = np.searchsorted(lib_nm, [target * (1 - ppm * 1e-6), target * (1 + ppm * 1e-6)])
    cand = np.arange(lo, hi)
    if mode == 'class1':
        cand = cand[~((lib_key[cand] == q['key']) & (lib_lib[cand] == q['lib']))]
    else:
        cand = cand[lib_key[cand] != q['key']]
    cand = cand[~np.isin(lib_idx[cand], idx)]
    if len(cand) == 0:
        return [], 0.0
    scores = {}
    for i in idx:
        qmz, qit = spec(i); qmode = meta.ionization_mode.values[i]
        for c in cand[lib_mode[cand] == qmode]:
            cmz, cit = spec(lib_idx[c]); s = cosine(qmz, qit, cmz, cit)
            k = lib_key[c]; scores[k] = max(scores.get(k, 0.0), s)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [k for k, _ in ranked[:topk]], (ranked[0][1] if ranked else 0.0)


pool_mass = pool.mass.values; pm_order = np.argsort(pool_mass); pm_sorted = pool_mass[pm_order]
pool_key = pool.ik14.values; pool_formula = pool.formula.values
form_of = dict(zip(pool_key, pool_formula))
pc_mass = np.load('/content/data/ds/casmi26-pubchem-tier/pc_mass.npy', mmap_mode='r')
res = []
for qi, q in Q.iterrows():
    idx = q['idx']; nms = nm_all[idx]; target = float(np.median(nms[np.isfinite(nms)]))
    lo, hi = np.searchsorted(pm_sorted, [target * (1 - 10e-6), target * (1 + 10e-6)])
    hits = pm_order[lo:hi]; keys = pool_key[hits]; forms = pool_formula[hits]
    tf = form_of.get(q['key'], '')
    plo, phi = np.searchsorted(pc_mass, [target * (1 - 10e-6), target * (1 + 10e-6)])
    r = dict(set=q['set'], key=q['key'], n_spec=len(idx), mass=target, n_pool=len(hits), truth_in_pool=bool((keys == q['key']).any()),
             n_same_formula=int((forms == tf).sum()), n_pubchem=int(phi - plo))
    for mode in ('class1', 'class2'):
        ranked, top = search(q, mode)
        r[f'rr_{mode}'] = next((1.0 / (i + 1) for i, k in enumerate(ranked) if k == q['key']), 0.0)
        r[f'top_{mode}'] = top
    res.append(r)
    if qi % 100 == 0:
        print(f'  {qi}/{len(Q)} {time.time()-t0:.0f}s', flush=True)
R = pd.DataFrame(res); R.to_pickle('/content/work/val_baseline_seed0.pkl')
pd.set_option('display.width', 200)
print(R.groupby('set').agg(n=('key', 'size'), mrr_c1=('rr_class1', 'mean'), mrr_c2=('rr_class2', 'mean'), top1_c1=('rr_class1', lambda s: (s == 1).mean()),
                           in_pool=('truth_in_pool', 'mean'), n_pool=('n_pool', 'median'), same_formula=('n_same_formula', 'median'), n_pubchem=('n_pubchem', 'median'),
                           spec=('n_spec', 'mean')))
print('class-1 top cosine by set / mode:'); print(R.groupby('set')[['top_class1', 'top_class2']].describe().T)
print('BASELINE DONE', f'{time.time()-t0:.0f}s')
