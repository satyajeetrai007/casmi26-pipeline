"""Build a compact cache of train.parquet for model training.

Outputs in /content/work/cache:
  spec_meta.parquet      one row per spectrum (same order as train.parquet)
  peaks_mz.npy  [N,K] f32  top-K peaks by intensity, sorted by m/z, zero padded
  peaks_int.npy [N,K] f16  normalised intensities (base peak = 1)
  n_peaks.npy   [N] int16
  structs.parquet        unique structures: inchikey14, smiles, formula, mono mass, struct_id
  fp_morgan4096.npy      [M,512] uint8 packed Morgan r2 4096 bits
  fp_maccs.npy           [M,21] uint8 packed MACCS (167 bits)
  fp_morgan_count.npy    [M,2048] uint8 Morgan r2 counts (clipped 255)
Electron-mass note: the hidden test had an electron-mass fix applied; the host said to subtract
0.000548 Da from positive-mode m/z of the Enveda data to match. Applied to enveda-180 and
enveda-np-examples (same Enveda pipeline), positive mode, peaks and precursor.
"""
import os, sys, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from multiprocessing import Pool

SRC = '/content/data/comp/train.parquet'
OUT = '/content/work/cache'
K = 128
REL_FLOOR = 0.001
ELECTRON = 0.000548
os.makedirs(OUT, exist_ok=True)

META_COLS = ['adduct', 'adduct_orig', 'ionization_mode', 'instrument_type', 'precursor_mz',
             'collision_energy_ev', 'collision_energy_orig_units', 'ingest_lib', 'inchikey14', 'molecular_formula',
             'num_peaks', 'precursor_error_ppm', 'base_peak_intensity', 'normalized_smiles']


def process_group(gi):
    pf = pq.ParquetFile(SRC)
    t = pf.read_row_group(gi, columns=META_COLS + ['ms2_mzs', 'ms2_normalized_intensities'])
    n = t.num_rows
    mz_col = t.column('ms2_mzs').combine_chunks()
    it_col = t.column('ms2_normalized_intensities').combine_chunks()
    mz_off = mz_col.offsets.to_numpy(); mz_val = mz_col.values.to_numpy(zero_copy_only=False).astype(np.float64)
    it_off = it_col.offsets.to_numpy(); it_val = it_col.values.to_numpy(zero_copy_only=False).astype(np.float64)
    prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
    mode = np.asarray(t.column('ionization_mode').to_pylist())
    lib = np.asarray(t.column('ingest_lib').to_pylist())
    fix = (mode == 'positive') & np.isin(lib, ['enveda-180', 'enveda-np-examples'])
    out_mz = np.zeros((n, K), np.float32); out_it = np.zeros((n, K), np.float16); npk = np.zeros(n, np.int16)
    for i in range(n):
        mz = mz_val[mz_off[i]:mz_off[i + 1]]; it = it_val[it_off[i]:it_off[i + 1]]
        if len(mz) == 0:
            continue
        if fix[i]:
            mz = mz - ELECTRON
        m = it.max()
        if m > 0 and m != 1.0:
            it = it / m
        keep = (it >= REL_FLOOR) & np.isfinite(mz) & (mz > 0)
        if np.isfinite(prec[i]):
            keep &= mz <= prec[i] + 2.0 + (ELECTRON if fix[i] else 0.0)
        mz = mz[keep]; it = it[keep]
        if len(mz) > K:
            idx = np.argpartition(-it, K - 1)[:K]; mz = mz[idx]; it = it[idx]
        o = np.argsort(mz); mz = mz[o]; it = it[o]
        out_mz[i, :len(mz)] = mz; out_it[i, :len(mz)] = it; npk[i] = len(mz)
    meta = t.select(META_COLS).to_pandas()
    meta['precursor_mz'] = np.where(fix, prec - ELECTRON, prec)
    ce = meta['collision_energy_ev']
    meta['ce_mean'] = ce.map(lambda v: float(np.mean(v)) if v is not None and len(v) else np.nan).astype(np.float32)
    meta['ce_n'] = ce.map(lambda v: len(v) if v is not None else 0).astype(np.int16)
    meta = meta.drop(columns=['collision_energy_ev'])
    meta['row_group'] = gi
    return gi, meta, out_mz, out_it, npk


def fp_worker(args):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    genc = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles_list = args
    bits = np.zeros((len(smiles_list), 512), np.uint8); maccs = np.zeros((len(smiles_list), 21), np.uint8)
    cnt = np.zeros((len(smiles_list), 2048), np.uint8); mass = np.full(len(smiles_list), np.nan, np.float64)
    formula = [''] * len(smiles_list); ok = np.zeros(len(smiles_list), bool); ik14 = [''] * len(smiles_list)
    for j, s in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            a = np.zeros(4096, np.uint8);
            a[list(gen.GetFingerprint(m).GetOnBits())] = 1; bits[j] = np.packbits(a)
            b = np.zeros(168, np.uint8); b[list(MACCSkeys.GenMACCSKeys(m).GetOnBits())] = 1; maccs[j] = np.packbits(b[1:])
            c = genc.GetCountFingerprintAsNumPy(m); cnt[j] = np.clip(c, 0, 255)
            mass[j] = Descriptors.ExactMolWt(m); formula[j] = rdMolDescriptors.CalcMolFormula(m); ok[j] = True
            try:
                ik14[j] = Chem.MolToInchiKey(m)[:14]
            except Exception:
                pass
        except Exception:
            pass
    return bits, maccs, cnt, mass, formula, ok, ik14


if __name__ == '__main__':
    t0 = time.time()
    pf = pq.ParquetFile(SRC); G = pf.num_row_groups; N = pf.metadata.num_rows
    print('rows', N, 'groups', G, flush=True)
    metas = [None] * G; MZ = np.lib.format.open_memmap(f'{OUT}/peaks_mz.npy', 'w+', np.float32, (N, K))
    IT = np.lib.format.open_memmap(f'{OUT}/peaks_int.npy', 'w+', np.float16, (N, K)); NP = np.zeros(N, np.int16)
    sizes = [pf.metadata.row_group(g).num_rows for g in range(G)]; starts = np.concatenate([[0], np.cumsum(sizes)])
    with Pool(int(os.environ.get('NPROC', '10'))) as pool:
        for gi, meta, omz, oit, npk in pool.imap_unordered(process_group, range(G)):
            s, e = starts[gi], starts[gi + 1]; assert e - s == len(meta)
            MZ[s:e] = omz; IT[s:e] = oit; NP[s:e] = npk; metas[gi] = meta
            if gi % 10 == 0:
                print(f'  group {gi}/{G} {time.time()-t0:.0f}s', flush=True)
    MZ.flush(); IT.flush(); np.save(f'{OUT}/n_peaks.npy', NP)
    meta = pd.concat(metas, ignore_index=True)
    structs = meta[['inchikey14', 'normalized_smiles', 'molecular_formula']].drop_duplicates('inchikey14').reset_index(drop=True)
    structs['struct_id'] = np.arange(len(structs))
    meta = meta.merge(structs[['inchikey14', 'struct_id']], on='inchikey14', how='left')
    meta.to_parquet(f'{OUT}/spec_meta.parquet')
    print('spectra', len(meta), 'structures', len(structs), f'{time.time()-t0:.0f}s', flush=True)
    smi = structs.normalized_smiles.tolist(); chunks = [smi[i:i + 2000] for i in range(0, len(smi), 2000)]
    bits, maccs, cnt, mass, formula, ok = [], [], [], [], [], []
    with Pool(int(os.environ.get('NPROC', '12'))) as pool:
        for r in pool.imap(fp_worker, chunks):
            bits.append(r[0]); maccs.append(r[1]); cnt.append(r[2]); mass.append(r[3]); formula += r[4]; ok.append(r[5])
    np.save(f'{OUT}/fp_morgan4096.npy', np.concatenate(bits)); np.save(f'{OUT}/fp_maccs.npy', np.concatenate(maccs))
    np.save(f'{OUT}/fp_morgan_count.npy', np.concatenate(cnt))
    structs['mono_mass'] = np.concatenate(mass); structs['formula_rdkit'] = formula; structs['rdkit_ok'] = np.concatenate(ok)
    structs.to_parquet(f'{OUT}/structs.parquet')
    print('fp done; rdkit failures', int((~structs.rdkit_ok).sum()), f'{time.time()-t0:.0f}s', flush=True)
    print('CACHE DONE')
