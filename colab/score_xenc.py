"""Add cross-encoder scores to candidate records (val_scores_c*.pkl / test_scores.pkl written by infer_fp.py).
Every candidate is featurised from its SMILES with molfeat.featurize (same function as training), scores are averaged
over the molecule's spectra, and stored as column 'xenc' (aligned with 'keys'). Output: same records + xenc.
usage: python score_xenc.py --ckpt runs/xenc_v1/ema.pt --mode val --scores val_scores_c1.pkl,val_scores_c2.pkl --outdir /kaggle/working
       python score_xenc.py --ckpt runs/xenc_v1/ema.pt --mode test --scores test_scores.pkl --outdir /content/work
"""
import os, time, argparse, numpy as np, pandas as pd, torch
from multiprocessing import Pool
from fpmodel import *
from xenc import XEnc
from molfeat import featurize, A_MAX, F_MAX

p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--mode', default='val'); p.add_argument('--scores', required=True)
p.add_argument('--outdir', required=True); p.add_argument('--chunk', type=int, default=96); p.add_argument('--nproc', type=int, default=4)
p.add_argument('--test_path', default='/content/data/comp/test.parquet')
args = p.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
AMP = torch.bfloat16 if (dev == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
ck = torch.load(args.ckpt, map_location='cpu'); sa = ck['spec_args']; xa = ck['args']
model = XEnc(SpecModel(d=sa.get('d', 512), layers=sa.get('layers', 6)), spec_d=sa.get('d', 512), d=xa['d'], layers=xa['layers'])
model.load_state_dict(ck['model']); model.to(dev).eval()
CACHE = '/content/work/cache'

# ---- spectra per record id
if args.mode == 'val':
    meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet', columns=['adduct', 'ingest_lib', 'instrument_type', 'ionization_mode', 'ce_mean', 'ce_n', 'precursor_mz'])
    MZ = np.load(f'{CACHE}/peaks_mz.npy', mmap_mode='r'); IT = np.load(f'{CACHE}/peaks_int.npy', mmap_mode='r'); NPK = np.load(f'{CACHE}/n_peaks.npy')
    Q = pd.read_pickle('/content/work/val_queries_seed0.pkl'); qidx = dict(zip(Q['set'] + '|' + Q['key'], Q['idx']))
    def spectra_of(rid):
        idx = np.asarray(qidx[rid]); mz = MZ[idx].astype(np.float32); it = IT[idx].astype(np.float32); npk = NPK[idx].astype(np.int64)
        Mt = encode_meta(meta.adduct.values[idx], meta.ingest_lib.values[idx], meta.instrument_type.values[idx], meta.ionization_mode.values[idx], meta.ce_mean.values[idx], meta.ce_n.values[idx], meta.precursor_mz.values[idx])
        return mz, it, npk, Mt
else:
    te = pd.read_parquet(args.test_path); groups = {str(k): g for k, g in te.groupby('molecule_id', sort=False)}
    def spectra_of(rid):
        g = groups[str(rid)]; B = len(g); mz = np.zeros((B, K), np.float32); it = np.zeros((B, K), np.float32); npk = np.zeros(B, np.int64); ce_m, ce_n = [], []
        for b, r in enumerate(g.itertuples()):
            m = np.asarray(r.ms2_mzs, np.float64); i = np.asarray(r.ms2_normalized_intensities, np.float64)
            keep = (i >= 0.001) & (m <= r.precursor_mz + 2.0); m, i = m[keep], i[keep]
            if len(m) > K:
                sel = np.argpartition(-i, K - 1)[:K]; m, i = m[sel], i[sel]
            o = np.argsort(m); m, i = m[o], i[o]
            if len(i) and i.max() > 0: i = i / i.max()
            n = len(m); mz[b, :n] = m; it[b, :n] = i; npk[b] = n
            ce = np.atleast_1d(np.asarray(r.collision_energy_ev, float)) if r.collision_energy_ev is not None else np.zeros(0)
            ce_m.append(float(ce.mean()) if len(ce) else np.nan); ce_n.append(len(ce))
        Mt = encode_meta(g.adduct.values, ['enveda-np-examples'] * B, g.instrument_type.values, g.ionization_mode.values, ce_m, ce_n, g.precursor_mz.values)
        return mz, it, npk, Mt


def _feat(s):
    f = featurize(s)
    return None if f is None else (f['atoms'], f['frags'], f['cnt'], (f['n_atoms'], f['n_frags']))


for path in args.scores.split(','):
    R = pd.read_pickle(path); name = os.path.basename(path)
    smis = sorted({s for lst in R['smiles'] if isinstance(lst, list) for s in lst})
    with Pool(args.nproc) as pp:
        feats = dict(zip(smis, pp.map(_feat, smis, chunksize=256)))
    print(name, 'records', len(R), 'unique candidates', len(smis), 'featurised', sum(v is not None for v in feats.values()), f'{time.time()-t0:.0f}s', flush=True)
    out = []
    with torch.no_grad():
        for qi, r in R.iterrows():
            smil = r['smiles'] if isinstance(r.get('smiles'), list) else []
            sc = np.full(len(smil), np.nan, np.float32)
            if len(smil) == 0:
                out.append(sc); continue
            mz, it, npk, Mt = spectra_of(r['id']); S_ = len(npk)
            st = dict(mz=torch.from_numpy(mz).to(dev), it=torch.from_numpy(it).to(dev), npk=torch.from_numpy(npk).to(dev))
            for k_ in ('adduct', 'lib', 'instr', 'mode', 'cef', 'prec'): st[k_] = torch.from_numpy(Mt[k_]).to(dev)
            good = [j for j, s in enumerate(smil) if feats.get(s) is not None]
            for c0 in range(0, len(good), args.chunk):
                js = good[c0:c0 + args.chunk]; G = len(js)
                A = np.stack([feats[smil[j]][0] for j in js]); Fr = np.stack([feats[smil[j]][1] for j in js]); C = np.stack([feats[smil[j]][2] for j in js]); N = np.array([feats[smil[j]][3] for j in js], np.int16)
                ct = dict(atoms=torch.from_numpy(np.broadcast_to(A, (S_,) + A.shape).copy()).to(dev), frags=torch.from_numpy(np.broadcast_to(Fr, (S_,) + Fr.shape).copy()).to(dev),
                          nn_=torch.from_numpy(np.broadcast_to(N, (S_,) + N.shape).copy()).to(dev), cnt=torch.from_numpy(np.broadcast_to(C, (S_,) + C.shape).copy()).to(dev))
                with torch.autocast(dev, dtype=AMP, enabled=dev == 'cuda'):
                    s_ = model(st['mz'], st['it'], st['npk'], st['adduct'], st['lib'], st['instr'], st['mode'], st['cef'], st['prec'], ct['atoms'], ct['frags'], ct['nn_'], ct['cnt']).float().mean(0)
                sc[js] = s_.cpu().numpy()
            out.append(sc)
            if qi % 300 == 0: print(f'  {name} {qi}/{len(R)} {time.time()-t0:.0f}s', flush=True)
    R['xenc'] = out; R.to_pickle(os.path.join(args.outdir, name))
    if 'rr' in R.columns:
        def rr(r):
            x = np.nan_to_num(np.asarray(r['xenc'], np.float32), nan=-1e9); order = np.argsort(-x)[:25]
            return next((1.0 / (i + 1) for i, j in enumerate(order) if r['keys'][j] == r['key']), 0.0)
        R['rr_xenc'] = R.apply(rr, axis=1); print(name, 'xenc-only MRR@25 by set:', R.groupby('set').rr_xenc.mean().round(4).to_dict(), '| fused-order MRR:', R.groupby('set').rr.mean().round(4).to_dict())
print('SCORE XENC DONE', f'{time.time()-t0:.0f}s')
