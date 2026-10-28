"""Score candidates for query molecules with the FP model + cosine library matching + PubChem window channel.
Candidates = pool (train + COCONUT + PubChemLite + ChEBI/LMSD) within --ppm, plus up to --pubchem_top PubChem-only
structures from the same mass window (most documented first). Writes per-candidate records for the ranker.
usage: python infer_fp.py --ckpt runs/fp_v1/ema.pt --mode test --out /kaggle/working/test_scores.pkl --sub /kaggle/working/submission.csv
       python infer_fp.py --ckpt runs/fp_v1/ema.pt --mode val --out /kaggle/working/val_scores.pkl
"""
import os, sys, glob, time, argparse, pickle, numpy as np, pandas as pd, torch
from fpmodel import *
from build_train_cache import fp_worker

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True); p.add_argument('--mode', default='test'); p.add_argument('--out', required=True); p.add_argument('--sub', default='')
p.add_argument('--val_queries', default='/content/work/val_queries_seed0.pkl'); p.add_argument('--ppm', type=float, default=10.0)
p.add_argument('--pubchem_top', type=int, default=500); p.add_argument('--w_con', type=float, default=1.0); p.add_argument('--keep', type=int, default=150)
p.add_argument('--test_path', default='/content/data/comp/test.parquet'); p.add_argument('--nproc', type=int, default=3); p.add_argument('--frag_top', type=int, default=40)
p.add_argument('--limit', type=int, default=0); p.add_argument('--class2', action='store_true', help='val: remove ALL library spectra of the truth structure (class-2 simulation)')
p.add_argument('--truth_pubchem', action='store_true', help='val: also remove the truth from the pool so it can only re-enter through the PubChem window (test-like class 2)')
p.add_argument('--pc_select', default='fp', choices=['fp', 'pop']); p.add_argument('--pc_pop_union', type=int, default=100); p.add_argument('--pc_cap', type=int, default=20000)
p.add_argument('--sub_score', default='fused', choices=['fused', 'll', 'con', 'prior'], help='submission base score: fused, fingerprint log-lik only, contrastive only, or popularity prior only (spectrum-blind)')
p.add_argument('--analog', type=int, default=0, help='mass-shifted analog search: number of analogs per molecule (0 = off)'); p.add_argument('--w_ap', type=float, default=0.0, help='fused += w_ap * z(analog propagation ap)')
p.add_argument('--analog_win', type=float, default=200.0)
p.add_argument('--np_lib_as', default='', help="encode spectra whose library token is 'enveda-np-examples' as this library instead (its embedding is untrained when S1 is held out)")
p.add_argument('--lib_gate', type=float, default=0.0, help='submission order: library matches with cosine >= gate go first (0 = off)'); p.add_argument('--pop_w', type=float, default=0.0, help='submission order: + pop_w * (ln(1+SID) + ln(1+PMID))')
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; t0 = time.time()
CACHE = '/content/work/cache'; POOLS = '/content/work/pools'
AMP = torch.bfloat16 if (dev == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
TEST_ADDUCTS = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']
POPCOLS = ['lsid', 'lpmid', 'lpatent', 'np', 'n_vendors', 'lotus_refs', 'lmesh', 'lsynonyms', 'n_depositors', 'src_mask', 'formula_isomers']

# ---- model
ck = torch.load(args.ckpt, map_location=dev); a = ck['args']
model = SpecModel(d=a['d'], layers=a['layers']).to(dev); model.load_state_dict(ck['model']); model.eval()
molenc = MolEncoder(emb=512).to(dev); molenc.load_state_dict(ck['molenc']); molenc.eval()

# ---- pool
pool = pd.read_parquet(f'{POOLS}/pool.parquet'); P_BITS = np.load(f'{POOLS}/fp_morgan4096.npy'); P_MACCS = np.load(f'{POOLS}/fp_maccs.npy'); P_CNT = np.load(f'{POOLS}/fp_morgan_count.npy', mmap_mode='r')
pool_mass = pool.mass.values; pm_order = np.argsort(pool_mass); pm_sorted = pool_mass[pm_order]; pool_key = pool.ik14.values; pool_smiles = pool.smiles.values
pool_formula = pool.formula.values; pool_src = pool.src_mask.values.astype(np.int16); pool_row = {k: i for i, k in enumerate(pool_key)}
pp_path = sorted(glob.glob('/kaggle/input/**/pool_pop.parquet', recursive=True) + glob.glob('/content/work/pop/pool_pop.parquet'), key=len)
POOL_POP = pd.read_parquet(pp_path[0]) if pp_path else None
print('pool', len(pool), 'pool_pop', pp_path[:1], flush=True)
# ---- PubChem tier + popularity arrays (index-aligned)
PC = os.path.dirname(sorted(glob.glob('/kaggle/input/**/casmi26-pubchem-tier/pc_smiles.npy', recursive=True) + glob.glob('/content/data/ds/casmi26-pubchem-tier/pc_smiles.npy'), key=len)[0])
pc_mass = np.load(f'{PC}/pc_mass.npy', mmap_mode='r'); pc_off = np.load(f'{PC}/pc_off.npy', mmap_mode='r'); pc_smi = np.load(f'{PC}/pc_smiles.npy', mmap_mode='r')
popdir = sorted(glob.glob('/kaggle/input/**/pc_ik14.npy', recursive=True), key=len)
PCPOP = {}
if popdir and args.pubchem_top > 0:
    popdir = os.path.dirname(popdir[0]); pc_keys = np.load(f'{popdir}/pc_ik14.npy', mmap_mode='r')
    for n in POPCOLS:
        f = f'{popdir}/pc_{n}.npy'
        if os.path.exists(f):
            PCPOP[n] = np.load(f, mmap_mode='r')
    assert len(pc_keys) == len(pc_mass), (len(pc_keys), len(pc_mass))
    print('pubchem popularity columns', list(PCPOP), 'keys dtype', pc_keys.dtype, flush=True)
else:
    pc_keys = None; args.pubchem_top = 0


def key_str(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in 'SU':
        return arr.astype(str)
    return np.ascontiguousarray(arr).view('S14').reshape(-1).astype(str)


# ---- spectral library (for class-1 cosine matching)
meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet', columns=['inchikey14', 'adduct', 'precursor_mz', 'ionization_mode', 'ingest_lib', 'ce_mean', 'ce_n', 'instrument_type'])
MZ = np.load(f'{CACHE}/peaks_mz.npy', mmap_mode='r'); IT = np.load(f'{CACHE}/peaks_int.npy', mmap_mode='r'); NPK = np.load(f'{CACHE}/n_peaks.npy')
shift_all = np.array([ADDUCT_SHIFT.get(x, np.nan) for x in meta.adduct.values]); nm_all = meta.precursor_mz.values - shift_all
lib_ok = meta.adduct.isin(TEST_ADDUCTS).values & np.isfinite(nm_all) & (NPK >= 3)
lib_idx = np.where(lib_ok)[0]; o = np.argsort(nm_all[lib_idx]); lib_idx = lib_idx[o]; lib_nm = nm_all[lib_idx]
lib_key = meta.inchikey14.values[lib_idx]; lib_lib = meta.ingest_lib.values[lib_idx]; lib_mode = meta.ionization_mode.values[lib_idx]
print('library', len(lib_idx), f'{time.time()-t0:.0f}s', flush=True)
AIDX = None
if args.analog > 0:
    import analog as anmod
    AIDX = anmod.AnalogIndex(MZ, IT, NPK, meta.inchikey14.values.astype(str), meta.ionization_mode.values.astype(str), meta.ingest_lib.values.astype(str), nm_all, lib_ok)
    print('analog index', len(AIDX.rows), 'representatives', f'{time.time()-t0:.0f}s', flush=True)
W_SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def spec_arrays(i):
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


def lib_search(qspecs, target, exclude_key=None, exclude_lib=None, exclude_idx=()):
    lo, hi = np.searchsorted(lib_nm, [target * (1 - args.ppm * 1e-6), target * (1 + args.ppm * 1e-6)]); cand = np.arange(lo, hi)
    if exclude_key is not None:
        cand = cand[~((lib_key[cand] == exclude_key) & ((lib_lib[cand] == exclude_lib) if exclude_lib else True))]
    cand = cand[~np.isin(lib_idx[cand], exclude_idx)]
    scores = {}
    for qs in qspecs:
        qmz, qit = qs['mz'], np.sqrt(qs['it'])
        for c in cand[lib_mode[cand] == qs['mode']]:
            cmz, cit = spec_arrays(lib_idx[c]); s = cosine(qmz, qit, cmz, cit); k = lib_key[c]; scores[k] = max(scores.get(k, 0.0), s)
    return scores


# ---- queries
def test_queries():
    te = pd.read_parquet(args.test_path); qs = []
    for mid, sub in te.groupby('molecule_id', sort=False):
        specs = []
        for r in sub.itertuples():
            mz = np.asarray(r.ms2_mzs, np.float64); it = np.asarray(r.ms2_normalized_intensities, np.float64)
            keep = (it >= 0.001) & (mz <= r.precursor_mz + 2.0); mz, it = mz[keep], it[keep]
            if len(mz) > K:
                sel = np.argpartition(-it, K - 1)[:K]; mz, it = mz[sel], it[sel]
            o = np.argsort(mz); mz, it = mz[o], it[o]
            if len(it) and it.max() > 0: it = it / it.max()
            ce = np.atleast_1d(np.asarray(r.collision_energy_ev, float)) if r.collision_energy_ev is not None else np.zeros(0)
            specs.append(dict(mz=mz, it=it, adduct=r.adduct, mode=r.ionization_mode, lib='enveda-np-examples', instr=r.instrument_type,
                              ce_mean=float(ce.mean()) if len(ce) else np.nan, ce_n=len(ce), prec=float(r.precursor_mz)))
        qs.append(dict(id=mid, specs=specs, key=None, exclude=()))
    return qs


def val_queries():
    Q = pd.read_pickle(args.val_queries); qs = []
    for _, q in Q.iterrows():
        specs = []
        for i in q['idx']:
            n = int(NPK[i]); specs.append(dict(mz=np.asarray(MZ[i, :n], np.float64), it=np.asarray(IT[i, :n], np.float64), adduct=meta.adduct.values[i], mode=meta.ionization_mode.values[i],
                                             lib=meta.ingest_lib.values[i], instr=meta.instrument_type.values[i], ce_mean=float(meta.ce_mean.values[i]), ce_n=int(meta.ce_n.values[i]), prec=float(meta.precursor_mz.values[i])))
        qs.append(dict(id=f"{q['set']}|{q['key']}", specs=specs, key=q['key'], lib=q['lib'], exclude=np.asarray(q['idx']), set=q['set']))
    return qs


queries = test_queries() if args.mode == 'test' else val_queries()
if args.np_lib_as:
    n_sw = 0
    for q_ in queries:
        for s_ in q_['specs']:
            if s_['lib'] == 'enveda-np-examples':
                s_['lib'] = args.np_lib_as; n_sw += 1
    print('library token enveda-np-examples ->', args.np_lib_as, 'for', n_sw, 'spectra')
if args.limit:
    queries = queries[:args.limit]
print('queries', len(queries), f'{time.time()-t0:.0f}s', flush=True)


@torch.no_grad()
def predict(specs):
    B = len(specs); mz = np.zeros((B, K), np.float32); it = np.zeros((B, K), np.float32); npk = np.zeros(B, np.int64)
    for b, s in enumerate(specs):
        n = min(len(s['mz']), K); mz[b, :n] = s['mz'][:n]; it[b, :n] = s['it'][:n]; npk[b] = n
    M = encode_meta([s['adduct'] for s in specs], [s['lib'] for s in specs], [s['instr'] for s in specs], [s['mode'] for s in specs],
                    [s['ce_mean'] for s in specs], [s['ce_n'] for s in specs], [s['prec'] for s in specs])
    t = {k: torch.from_numpy(M[k]).to(dev) for k in ('adduct', 'lib', 'instr', 'mode', 'cef', 'prec')}
    with torch.autocast(dev, dtype=AMP, enabled=dev == 'cuda'):
        o = model(torch.from_numpy(mz).to(dev), torch.from_numpy(it).to(dev), torch.from_numpy(npk).to(dev), t)
    return torch.sigmoid(o['bits'].float()), torch.sigmoid(o['maccs'].float()), o['emb'].float(), (o['el'].float() * 10).mean(0).cpu().numpy()


@torch.no_grad()
def score_arrays(pb, pm, e, bits_packed, maccs_packed, cnt):
    yb = torch.from_numpy(np.unpackbits(bits_packed, axis=1)[:, :4096].astype(np.float32)).to(dev)
    ym = torch.from_numpy(np.unpackbits(maccs_packed, axis=1)[:, :167].astype(np.float32)).to(dev)
    yc = torch.from_numpy(np.asarray(cnt).astype(np.float32)).to(dev)
    ll = (fp_loglik(pb, yb) + fp_loglik(pm, ym)).mean(0)
    with torch.autocast(dev, dtype=AMP, enabled=dev == 'cuda'):
        me = molenc(yc, ym).float()
    con = (e @ me.T).mean(0)
    return ll.cpu().numpy(), con.cpu().numpy(), yb


def fp_bits_worker(smiles_list):
    """Morgan r2 4096 packed bits only (fast pre-screen of a whole PubChem window)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*'); gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    out = np.zeros((len(smiles_list), 512), np.uint8); ok = np.zeros(len(smiles_list), bool)
    for j, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None: continue
            out[j] = np.packbits(gen.GetFingerprintAsNumPy(m).astype(np.uint8)); ok[j] = True
        except Exception:
            pass
    return out, ok


def pubchem_window(target, exclude_keys, pb=None):
    """PubChem-only candidates of the mass window. --pc_select fp (default): pre-score up to --pc_cap window members by
    fingerprint log-likelihood and keep the best (pubchem_top - pc_pop_union) plus the pc_pop_union most documented;
    --pc_select pop: the pubchem_top most documented (V1 behaviour)."""
    lo, hi = np.searchsorted(pc_mass, [target * (1 - args.ppm * 1e-6), target * (1 + args.ppm * 1e-6)])
    if hi <= lo:
        return None
    keys = key_str(pc_keys[lo:hi]); idx = np.arange(lo, hi)
    keep = np.array([k not in exclude_keys for k in keys]); idx, keys = idx[keep], keys[keep]
    if len(idx) == 0:
        return None
    pop = np.zeros(len(idx), np.float32)
    for n in ('lsid', 'lpmid', 'lpatent'):
        if n in PCPOP: pop += np.asarray(PCPOP[n][idx]).astype(np.float32)
    if args.pc_select == 'fp' and pb is not None and len(idx) > args.pubchem_top:
        if len(idx) > args.pc_cap:                                   # very large windows: keep the most documented cap
            o = np.argsort(-pop, kind='stable')[:args.pc_cap]; idx, keys, pop = idx[o], keys[o], pop[o]
        smi_all = [bytes(pc_smi[pc_off[i]:pc_off[i + 1]]).decode() for i in idx]
        fr = POOL_MP.map(fp_bits_worker, [smi_all[i:i + 400] for i in range(0, len(smi_all), 400)])
        bits = np.concatenate([x[0] for x in fr]); okb = np.concatenate([x[1] for x in fr])
        ll = np.full(len(idx), -1e9, np.float32)
        with torch.no_grad():
            for c0 in range(0, len(idx), 8192):
                yb = torch.from_numpy(np.unpackbits(bits[c0:c0 + 8192], axis=1).astype(np.float32)).to(dev)
                ll[c0:c0 + 8192] = fp_loglik(pb, yb).mean(0).cpu().numpy()
        ll[~okb] = -1e9
        n_fit = max(0, args.pubchem_top - args.pc_pop_union)
        sel = list(dict.fromkeys(list(np.argsort(-ll, kind='stable')[:n_fit]) + list(np.argsort(-pop, kind='stable')[:args.pc_pop_union])))
        sel = np.array(sel[:args.pubchem_top]); idx, keys = idx[sel], keys[sel]
        smiles = [smi_all[i] for i in sel]
    else:
        sel = np.argsort(-pop, kind='stable')[:args.pubchem_top]; idx, keys = idx[sel], keys[sel]
        smiles = [bytes(pc_smi[pc_off[i]:pc_off[i + 1]]).decode() for i in idx]
    feats = {n: np.asarray(PCPOP[n][idx]).astype(np.float32) for n in PCPOP}
    return dict(idx=idx, keys=keys, smiles=smiles, feats=feats, n_window=int(hi - lo))


def _frag_job(a):
    smi, specs = a
    try:
        from fragscore import frag_score
        return frag_score(smi, specs)
    except Exception:
        return (np.nan, np.nan, np.nan, 0)


import multiprocessing as mp
POOL_MP = mp.get_context('fork').Pool(args.nproc)
results = []
for qi, q in enumerate(queries):
    specs = q['specs']; nms = np.array([s['prec'] - ADDUCT_SHIFT.get(s['adduct'], np.nan) for s in specs]); target = float(np.nanmedian(nms))
    lo, hi = np.searchsorted(pm_sorted, [target * (1 - args.ppm * 1e-6), target * (1 + args.ppm * 1e-6)]); cidx = pm_order[lo:hi]
    if args.truth_pubchem and q['key'] is not None:
        cidx = cidx[pool_key[cidx] != q['key']]
    r = dict(id=q['id'], set=q.get('set'), key=q['key'], mass=target, n_pool=len(cidx), n_spec=len(specs), truth_pubchem=bool(args.truth_pubchem), pc_select=args.pc_select)
    libs = lib_search(specs, target, exclude_key=q['key'], exclude_lib=(None if args.class2 else q.get('lib')), exclude_idx=q['exclude']) if q['key'] is not None else lib_search(specs, target)
    r['class2'] = bool(args.class2)
    r['lib_top'] = max(libs.values()) if libs else 0.0
    pb, pm, e, el = predict(specs); r['el_pred'] = el
    # --- candidate records: pool first, then PubChem window
    keys = list(pool_key[cidx]); smiles = list(pool_smiles[cidx]); formula = list(pool_formula[cidx]); mass = list(pool_mass[cidx]); src = list(pool_src[cidx])
    bits_p = P_BITS[cidx]; maccs_p = P_MACCS[cidx]; cnt_p = np.asarray(P_CNT[cidx])
    popf = {n: np.zeros(len(cidx), np.float32) for n in POPCOLS}
    if POOL_POP is not None and len(cidx):
        for n in POPCOLS:
            col = {'np': 'np_flag'}.get(n, n)
            if col in POOL_POP.columns: popf[n] = POOL_POP[col].values[cidx].astype(np.float32)
    r['n_pubchem'] = 0
    if args.pubchem_top > 0:
        w = pubchem_window(target, set(keys), pb)
        if w is not None and len(w['keys']):
            r['n_pubchem'] = len(w['keys']); r['n_pubchem_window'] = w['n_window']
            chunks = [w['smiles'][i:i + 100] for i in range(0, len(w['smiles']), 100)]
            fr = POOL_MP.map(fp_worker, chunks)
            wb = np.concatenate([x[0] for x in fr]); wm = np.concatenate([x[1] for x in fr]); wc = np.concatenate([x[2] for x in fr]); wmass = np.concatenate([x[3] for x in fr]); wform = sum([x[4] for x in fr], []); wok = np.concatenate([x[5] for x in fr])
            good = np.where(wok)[0]
            keys += [w['keys'][i] for i in good]; smiles += [w['smiles'][i] for i in good]; formula += [wform[i] for i in good]; mass += [float(wmass[i]) for i in good]; src += [16] * len(good)
            bits_p = np.concatenate([bits_p, wb[good]]); maccs_p = np.concatenate([maccs_p, wm[good]]); cnt_p = np.concatenate([cnt_p, wc[good]])
            for n in POPCOLS:
                popf[n] = np.concatenate([popf[n], w['feats'][n][good] if n in w['feats'] else np.zeros(len(good), np.float32)])
    n_all = len(keys)
    if n_all == 0:
        r.update(keys=[], smiles=[], n_cands=0); results.append(r); continue
    keys = np.array(keys); src = np.array(src, np.int16)
    ll, con, yb = score_arrays(pb, pm, e, bits_p, maccs_p, cnt_p)
    z = lambda s: (s - s.mean()) / (s.std() + 1e-6) if len(s) > 1 else s * 0
    fused = z(ll) + args.w_con * z(con)
    apf = None
    if AIDX is not None:
        an = AIDX.search(specs, target, win=args.analog_win, n_out=args.analog, exclude_key=q['key'], exclude_lib=(None if args.class2 else q.get('lib'))) if q['key'] is not None else AIDX.search(specs, target, win=args.analog_win, n_out=args.analog)
        an = [(k, v, sh) for k, v, sh in an if k in pool_row]
        an_bits = np.unpackbits(P_BITS[np.array([pool_row[k] for k, _, _ in an], np.int64)], axis=1)[:, :4096].astype(np.float32) if an else np.zeros((0, 4096), np.float32)
        apf = anmod.analog_features(yb.cpu().numpy() if torch.is_tensor(yb) else yb, an_bits, np.array([v for _, v, _ in an], np.float32))
        r['an_top_sim'] = float(an[0][1]) if an else 0.0; r['an_n'] = len(an)
        if q['key'] is not None:
            ti = np.where(keys == q['key'])[0]
            sw = {}
            for w in W_SWEEP:
                for base, bn in ((fused, 'fused'), (z(ll), 'll')):
                    sc_ = base + w * z(apf['ap'])
                    sw[f'{bn}+{w}'] = (1.0 / (1 + int((sc_ > sc_[ti[0]]).sum()))) if len(ti) else 0.0
            r['rr_sweep'] = {k_: (v_ if v_ >= 1 / 25 else 0.0) for k_, v_ in sw.items()}
        if args.w_ap > 0:
            fused = fused + args.w_ap * z(apf['ap'])
    libsc = np.array([libs.get(k, 0.0) for k in keys], np.float32)
    # analog propagation: Tanimoto of each candidate to the structures of the best library hits (train structures)
    hits = sorted(((v, k) for k, v in libs.items() if v >= 0.3 and k in pool_row), reverse=True)[:8]
    analog = np.zeros(n_all, np.float32); analog_top = np.zeros(n_all, np.float32)
    if hits:
        hrows = np.array([pool_row[k] for _, k in hits]); hcos = np.array([v for v, _ in hits], np.float32)
        hb = torch.from_numpy(np.unpackbits(P_BITS[hrows], axis=1)[:, :4096].astype(np.float32)).to(dev)
        inter = yb @ hb.T; union = yb.sum(1)[:, None] + hb.sum(1)[None, :] - inter; tan = (inter / union.clamp(min=1)).cpu().numpy()
        same = keys[:, None] == np.array([k for _, k in hits])[None, :]; tan_excl = np.where(same, 0.0, tan)
        analog = (tan_excl * hcos[None, :]).max(1).astype(np.float32); analog_top = tan_excl[:, 0].astype(np.float32)
    r['n_hits'] = len(hits)
    # keep top --keep by fused score, but never fewer than 60 pool candidates and 60 PubChem candidates
    order = np.argsort(-fused, kind='stable'); is_pc = src == 16
    keep = set(order[:args.keep].tolist()) | set(order[~is_pc[order]][:60].tolist()) | set(order[is_pc[order]][:60].tolist())
    order = np.array([i for i in order if i in keep])
    rec = dict(keys=keys[order].tolist(), smiles=[smiles[i] for i in order], formula=[formula[i] for i in order], mass=np.array(mass)[order], src=src[order],
               ll=ll[order], con=con[order], fused=fused[order], libsc=libsc[order], analog=analog[order], analog_top=analog_top[order], n_cands=n_all)
    if apf is not None:
        for n_, v_ in apf.items():
            rec[n_] = v_[order]
    for n in POPCOLS:
        rec['pop_' + n] = popf[n][order]
    if args.frag_top > 0:
        sp = [dict(mz=s['mz'], it=s['it'], adduct=s['adduct'], mode=s['mode']) for s in specs]
        fr = POOL_MP.map(_frag_job, [(rec['smiles'][i], sp) for i in range(min(args.frag_top, len(order)))], chunksize=4)
        fw = np.full(len(order), np.nan, np.float32); fe = fw.copy(); fp_ = fw.copy()
        fw[:len(fr)] = [x[0] for x in fr]; fe[:len(fr)] = [x[1] for x in fr]; fp_[:len(fr)] = [x[2] for x in fr]
        rec.update(frag_w=fw, frag_e=fe, frag_p=fp_)
    r.update(rec)
    if q['key'] is not None:
        r['rr'] = next((1.0 / (i + 1) for i, k in enumerate(rec['keys'][:25]) if k == q['key']), 0.0)
        r['truth_rank'] = next((i + 1 for i, k in enumerate(rec['keys']) if k == q['key']), 0)
    results.append(r)
    if qi % 50 == 0:
        print(f'  {qi}/{len(queries)} {time.time()-t0:.0f}s', flush=True)
R = pd.DataFrame(results); R.to_pickle(args.out)
if args.mode == 'val':
    print(R.groupby('set').agg(mrr=('rr', 'mean'), top1=('rr', lambda s: (s == 1).mean()), in_list=('truth_rank', lambda s: (s > 0).mean()), n_pool=('n_pool', 'median'), n_pc=('n_pubchem', 'median'), n=('rr', 'size')))
    if 'rr_sweep' in R.columns:
        SW = pd.DataFrame([dict(set=s_, **d_) for s_, d_ in zip(R['set'], R['rr_sweep']) if isinstance(d_, dict)])
        print('ANALOG SWEEP (MRR@25 over all candidates; base + w * z(ap))'); print(SW.groupby('set').mean().round(4).T.to_string())
        print('analog top sim median by set', R.groupby('set').an_top_sim.median().round(3).to_dict())
if args.sub:
    rows = []; n_gate = 0
    for r in results:
        seen, out = set(), []
        keys_, smis_ = list(r.get('keys', [])), list(r.get('smiles', []))
        if keys_:
            # submission order: fused spectral score + pop_w * (ln(1+substances) + ln(1+PubMed)) ; strong library matches first
            if args.sub_score == 'prior':
                sc = np.asarray(r['pop_lsid'], np.float64) + np.asarray(r['pop_lpmid'], np.float64)
            else:
                sc = np.asarray(r[args.sub_score], np.float64).copy()
            if args.pop_w > 0 and 'pop_lsid' in r:
                sc += args.pop_w * (np.asarray(r['pop_lsid'], np.float64) + np.asarray(r['pop_lpmid'], np.float64))
            if args.lib_gate > 0:
                ls = np.asarray(r['libsc'], np.float64); g = ls >= args.lib_gate; n_gate += int(g.any())
                sc = np.where(g, 1e6 + ls, sc)
            o = np.argsort(-sc, kind='stable'); keys_ = [keys_[i] for i in o]; smis_ = [smis_[i] for i in o]
        for k, s in zip(keys_, smis_):
            if k in seen: continue
            seen.add(k); out.append(s)
            if len(out) == 25: break
        rows.append(dict(molecule_id=r['id'], smiles=';'.join(out) if out else 'C'))
    sub = pd.DataFrame(rows); sub.to_csv(args.sub, index=False); print('written', args.sub, len(sub), '| molecules with a library match >= gate:', n_gate)
print('INFER DONE', f'{time.time()-t0:.0f}s')
