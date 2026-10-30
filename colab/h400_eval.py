"""Score answer files on the hard400 fixture (400 held-out training molecules, classes 1/2/3).
Identity = InChIKey-14 after RDKit tautomer canonicalisation (as the metric). Reports MRR@25 / top-1 / in-list per class,
and on the subset that our own model never trained on (our S1 validation structures) when --s1 is given.
usage: python h400_eval.py --fix DIR [--s1 val_queries_seed0.pkl] name=file.csv [name=file.csv ...]
"""
import argparse, functools, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()


@functools.lru_cache(maxsize=200000)
def ident(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        try:
            m = _TE.Canonicalize(m)
        except Exception:
            pass
        return Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def ranks(sub, truth):
    S = pd.read_csv(sub, dtype=str, keep_default_na=False).set_index('molecule_id').smiles
    out = {}
    for mid, ts in zip(truth.molecule_id, truth.truth_smiles):
        t = ident(ts); L = S.get(mid, '').split(';')[:25]
        out[mid] = next((i + 1 for i, s in enumerate(L) if s and ident(s) == t), 0)
    return pd.Series(out)


def table(rk, truth, s1):
    T = truth.set_index('molecule_id'); rr = rk.map(lambda r: 1.0 / r if r else 0.0)
    rows = {}
    for name, m in [('all', T.index), ('c1', T.index[T.eval_class == 1]), ('c2', T.index[T.eval_class == 2]), ('c3', T.index[T.eval_class == 3])] + \
                   ([('S1', T.index[T.index.isin(s1)]), ('S1c2', T.index[T.index.isin(s1) & (T.eval_class == 2)])] if s1 else []):
        rows[name] = dict(mrr=round(float(rr[m].mean()), 4), top1=round(float((rk[m] == 1).mean()), 3), inlist=round(float((rk[m] > 0).mean()), 3), n=len(m))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fix', required=True); p.add_argument('--s1', default=''); p.add_argument('--out', default='')
    p.add_argument('files', nargs='+'); a = p.parse_args()
    truth = pd.read_parquet(f'{a.fix}/truth.parquet')
    s1 = set()
    if a.s1:
        q = pd.read_pickle(a.s1); s1 = set(q.loc[q.set == 'S1', 'key'])
    res, RK = {}, {}
    for f in a.files:
        name, path = f.split('=', 1)
        RK[name] = ranks(path, truth); res[name] = table(RK[name], truth, s1)
    for name, t in res.items():
        print(f'{name:>22} ' + ' | '.join(f"{k} {v['mrr']:.4f} (top1 {v['top1']:.3f}, in {v['inlist']:.3f})" for k, v in t.items()), flush=True)
    if a.out:
        pd.DataFrame(RK).to_csv(a.out); json.dump(res, open(a.out.replace('.csv', '.json'), 'w'), indent=1)
