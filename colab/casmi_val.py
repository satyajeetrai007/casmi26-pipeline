"""Validation harness: query sets that behave like the hidden test, class simulation, MRR@25.

Sets (structure = InChIKey14):
  S1  all enveda-np-examples structures (timsTOF natural products; all have public spectra)
  S2  enveda-180 structures with NO public spectra (timsTOF, synthetic chemistry)
  S3  natural-product-library structures (riken/gnps/massbank/mona/msdial/spectraverse/masaryk) that are in COCONUT
      and not in S1 (public-instrument NPs)
Each query molecule = up to MAXQ spectra of the structure (test adducts only), drawn from ONE library
(S1/S2: the Enveda library; S3: one public library). Class simulation = what is removed from the spectral library:
  class1: the query library's spectra of that structure only (other libraries' spectra of it remain)
  class2: every spectrum of the structure
Candidate pools are untouched (truth is always in the pool for S1/S3 by construction; S2 truth is a train structure).
"""
import numpy as np, pandas as pd

CACHE = '/content/work/cache'
PROTON = 1.007276
ADDUCT_SHIFT = {'[M+H]+': PROTON, '[M+NH4]+': 18.033823, '[M-H2O+H]+': PROTON - 18.010565, '[M-2H2O+H]+': PROTON - 2 * 18.010565,
                '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -PROTON, '[M-H2O-H]-': -PROTON - 18.010565,
                '[M+CH2O2-H]-': 46.005479 - PROTON, '[M+Cl]-': 34.969402}
TEST_ADDUCTS = list(ADDUCT_SHIFT)
NP_LIBS = ['riken', 'gnps', 'massbank', 'mona', 'msdial', 'spectraverse', 'masaryk']


def neutral_mass(prec, adduct):
    return np.asarray(prec, float) - np.array([ADDUCT_SHIFT.get(a, np.nan) for a in adduct])


def load_meta():
    meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet')
    meta['idx'] = np.arange(len(meta))
    meta['n_peaks'] = np.load(f'{CACHE}/n_peaks.npy')
    return meta


def build_sets(meta, seed=0, n_s2=500, n_s3=600, maxq=6, coconut_keys=None):
    rng = np.random.default_rng(seed)
    ok = meta.adduct.isin(TEST_ADDUCTS) & (meta.n_peaks >= 3)
    pub = set(meta.loc[~meta.ingest_lib.isin(['enveda-180', 'enveda-np-examples']), 'inchikey14'])
    sets = {}
    s1 = sorted(set(meta.loc[(meta.ingest_lib == 'enveda-np-examples') & ok, 'inchikey14']))
    sets['S1'] = [(k, 'enveda-np-examples') for k in s1]
    e180 = meta.loc[(meta.ingest_lib == 'enveda-180') & ok, 'inchikey14']
    cand = sorted(set(e180) - pub); rng.shuffle(cand)
    sets['S2'] = [(k, 'enveda-180') for k in cand[:n_s2]]
    npl = meta.loc[meta.ingest_lib.isin(NP_LIBS) & ok, ['inchikey14', 'ingest_lib']].drop_duplicates()
    keys3 = set(npl.inchikey14) - set(s1)
    print('S3 debug: npl rows', len(npl), 'np-lib structures', len(keys3), 'coconut keys', 0 if coconut_keys is None else len(coconut_keys), flush=True)
    if coconut_keys is not None:
        ck = set(str(k)[:14] for k in coconut_keys); ov = keys3 & ck; print('S3 debug: overlap', len(ov), flush=True)
        if len(ov) >= 300:   # the public COCONUT file excludes train structures, so the overlap is normally 0
            keys3 = ov
    keys3 = sorted(keys3); rng.shuffle(keys3); keys3 = keys3[:n_s3]
    lib_of = npl.groupby('inchikey14').ingest_lib.apply(list).to_dict()
    sets['S3'] = [(k, rng.choice(lib_of[k])) for k in keys3]
    # draw query spectra
    queries = []
    for name, lst in sets.items():
        for k, lib in lst:
            rows = meta.loc[(meta.inchikey14 == k) & (meta.ingest_lib == lib) & ok]
            n = int(np.clip(rng.geometric(0.33), 1, maxq)) if len(rows) > 1 else 1   # median ~3 like the test
            take = rows.sample(min(n, len(rows)), random_state=int(rng.integers(1 << 31)))
            queries.append(dict(set=name, key=k, lib=lib, idx=take.idx.values))
    return pd.DataFrame(queries)


def library_mask(meta, q, mode):
    """boolean mask over meta rows allowed in the spectral library for query q."""
    if mode == 'class1':
        return ~((meta.inchikey14.values == q['key']) & (meta.ingest_lib.values == q['lib']))
    return meta.inchikey14.values != q['key']


def mrr_at(rank_lists, truths, k=25):
    rr = []
    for lst, t in zip(rank_lists, truths):
        r = 0.0
        for i, c in enumerate(lst[:k]):
            if c == t:
                r = 1.0 / (i + 1); break
        rr.append(r)
    return float(np.mean(rr)), np.array(rr)
