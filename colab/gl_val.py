"""Measure what GLACIER / ICEBERG same-formula re-scoring adds on top of the FP model's validation lists.
Input: val_scores.pkl from infer_fp.py --mode val (top-100 candidates per query with ll/con/keys).
usage: python gl_val.py --scores val_scores.pkl --gl_pkg <casmi26-glacier dir> --ice_pkg <casmi26-iceberg dir> --budget 3000 --top_n 40
"""
import os, sys, time, json, argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--scores', required=True); p.add_argument('--gl_pkg', required=True); p.add_argument('--ice_pkg', required=True)
p.add_argument('--budget', type=float, default=3000); p.add_argument('--top_n', type=int, default=40); p.add_argument('--site', default='/kaggle/tmp/ice_site')
p.add_argument('--wheels', default=''); p.add_argument('--run_ice', type=int, default=1); p.add_argument('--out', default='gl_val_out.pkl'); p.add_argument('--mode', default='val'); p.add_argument('--test_path', default='/content/data/comp/test.parquet')
args = p.parse_args()
sys.path.insert(0, args.ice_pkg); sys.path.insert(0, args.gl_pkg)
import fuse, gl_fuse
CACHE = '/content/work/cache'
R = pd.read_pickle(args.scores)
meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet', columns=['adduct', 'precursor_mz', 'ionization_mode', 'instrument_type', 'ce_mean', 'ce_n'])
MZ = np.load(f'{CACHE}/peaks_mz.npy', mmap_mode='r'); IT = np.load(f'{CACHE}/peaks_int.npy', mmap_mode='r'); NPK = np.load(f'{CACHE}/n_peaks.npy')
def spectra_val(idx):
    out = []
    for i in idx:
        if meta.ionization_mode.values[i] != 'positive' or meta.adduct.values[i] not in ('[M+H]+', '[M+Na]+'):
            continue
        n = int(NPK[i]); ce = meta.ce_mean.values[i]
        out.append(dict(mz=[float(x) for x in MZ[i, :n]], it=[float(x) for x in IT[i, :n]], prec=float(meta.precursor_mz.values[i]), adduct=meta.adduct.values[i],
                        mode='positive', instrument=meta.instrument_type.values[i], ce_ev=[float(ce)] if np.isfinite(ce) else None, ce_orig=None, ce_units='eV' if np.isfinite(ce) else None))
    return out


if args.mode == 'val':
    Q = pd.read_pickle('/content/work/val_queries_seed0.pkl'); qidx = dict(zip(Q['set'] + '|' + Q['key'], Q['idx']))
    spectra_of = lambda r: spectra_val(qidx[r['id']])
else:
    te = pd.read_parquet(args.test_path); by = {}
    for row in te.itertuples(index=False):
        by.setdefault(str(row.molecule_id), []).append(row)
    def spectra_of(r):
        out = []
        for row in by.get(str(r['id']), []):
            if row.ionization_mode != 'positive' or row.adduct not in ('[M+H]+', '[M+Na]+'):
                continue
            ce = [float(x) for x in row.collision_energy_ev] if row.collision_energy_ev is not None else None
            out.append(dict(mz=[float(x) for x in row.ms2_mzs], it=[float(x) for x in row.ms2_normalized_intensities], prec=float(row.precursor_mz), adduct=row.adduct,
                            mode='positive', instrument=row.instrument_type, ce_ev=ce, ce_orig=str(row.collision_energy_orig), ce_units=str(row.collision_energy_orig_units)))
        return out

items = []; info = {}
for _, r in R.iterrows():
    smis_all = list(r.get('smiles', []))
    if not smis_all:
        continue
    spectra = spectra_of(r)
    if not spectra:
        continue
    smis = smis_all[:args.top_n]; forms = list(r['formula'])[:args.top_n]
    cands = fuse.ice_candidates(smis, forms, top_n=args.top_n)
    if not cands:
        continue
    items.append(dict(mid=r['id'], spectra=spectra, cands=cands)); info[r['id']] = (smis, forms, list(r['keys'][:args.top_n]), r['key'])
print('molecules to score', len(items), 'cands total', sum(len(x['cands']) for x in items), flush=True)
wheels = args.wheels or None
t0 = time.time()
GL = gl_fuse.run_gl(args.gl_pkg, items, workdir='/kaggle/tmp/gl_work', device='cuda', budget_s=args.budget, site=args.site, wheels=wheels)
print('GL done', len(GL), f'{time.time()-t0:.0f}s', flush=True)
ICE = {}
if args.run_ice:
    t0 = time.time(); ICE = fuse.run_ice(args.ice_pkg, items, workdir='/kaggle/tmp/ice_work', device='cuda', budget_s=args.budget, site=args.site)
    print('ICE done', len(ICE), f'{time.time()-t0:.0f}s', flush=True)
pd.to_pickle(dict(GL=GL, ICE=ICE, info=info), args.out)


def mrr(order_fn):
    rows = []
    for _, r in R.iterrows():
        keys = list(r['keys']); truth = r['key']
        if r['id'] in info:
            smis, forms, k2, _ = info[r['id']]; scores = list(-np.arange(len(smis), dtype=float))
            order = order_fn(r['id'], smis, k2, scores, forms); keys = [k2[i] for i in order] + keys[len(k2):]
        rr = next((1.0 / (i + 1) for i, k in enumerate(keys[:25]) if k == truth), 0.0)
        rows.append(dict(set=r['set'], rr=rr, covered=r['id'] in GL))
    D = pd.DataFrame(rows); return D.groupby('set').rr.mean().round(4).to_dict(), D[D.covered].groupby('set').rr.mean().round(4).to_dict()


if args.mode != 'val':
    print('GLVAL DONE'); sys.exit(0)
print('baseline (FP order)      ', mrr(lambda mid, s, k, sc, f: list(range(len(s)))))
for lam in (0.5, 1.0, 2.0):
    print(f'GL lam {lam}              ', mrr(lambda mid, s, k, sc, f, lam=lam: gl_fuse.rerank_multi(s, k, sc, f, [GL.get(mid, {})], [lam], top_n=args.top_n)))
    if ICE:
        print(f'ICE lam {lam}             ', mrr(lambda mid, s, k, sc, f, lam=lam: gl_fuse.rerank_multi(s, k, sc, f, [ICE.get(mid, {})], [lam], top_n=args.top_n)))
        print(f'GL+ICE lam {lam}          ', mrr(lambda mid, s, k, sc, f, lam=lam: gl_fuse.rerank_multi(s, k, sc, f, [GL.get(mid, {}), ICE.get(mid, {})], [lam, lam], top_n=args.top_n)))
print('GLVAL DONE')
