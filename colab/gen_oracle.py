"""Class-3 feasibility: how often is a validation truth one or two biosynthetic edits away from a relative?
A) spectral parents = the top mass-shifted analogs of the query (truth spectra hidden), as available at test time
B) oracle parents   = the training structures most Tanimoto-similar to the truth (upper bound)
usage: python gen_oracle.py --records v_c2_anl.pkl --out gen_oracle.pkl
"""
import argparse, time, numpy as np, pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import derive as dv

p = argparse.ArgumentParser(); p.add_argument('--records', required=True); p.add_argument('--out', default='gen_oracle.pkl')
p.add_argument('--n_parents', type=int, default=6); p.add_argument('--min_sim', type=float, default=0.3); p.add_argument('--steps', type=int, default=2)
p.add_argument('--pools', default='/content/work/pools'); p.add_argument('--workers', type=int, default=4)
args = p.parse_args()

pool = pd.read_parquet(f'{args.pools}/pool.parquet'); PB = np.load(f'{args.pools}/fp_morgan4096.npy', mmap_mode='r')
key = pool.ik14.values; smi = pool.smiles.values; mass = pool.mass.values.astype(np.float64); row = {k: i for i, k in enumerate(key)}
tr = np.where((pool.src_mask.values.astype(np.int64) & 1) > 0)[0]; tr = tr[np.argsort(mass[tr])]; trm = mass[tr]
o_ = np.argsort(tr); TRB = np.empty((len(tr), PB.shape[1]), np.uint8); TRB[o_] = np.asarray(PB[tr[o_]])
POP = np.array([bin(i).count('1') for i in range(256)], np.int32); trn = POP[TRB].sum(1)
print('pool', len(pool), 'train structures', len(tr), flush=True)
R = pd.read_pickle(args.records)


def ik(s):
    m = Chem.MolFromSmiles(s)
    try:
        return None if m is None else Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


def oracle_parents(k, n):
    i = row[k]; m = mass[i]; lo, hi = np.searchsorted(trm, [m - 330, m + 330])
    b = np.asarray(PB[i]); inter = POP[TRB[lo:hi] & b].sum(1); qn = POP[b].sum()
    tan = inter / (trn[lo:hi] + qn - inter + 1e-9); cand = tr[lo:hi]
    o = np.argsort(-tan); out = []
    for j in o:
        if key[cand[j]] != k:
            out.append((key[cand[j]], float(tan[j])))
        if len(out) >= n:
            break
    return out


def job(t):
    rid, truth_k, truth_smi, M, parents = t
    tk = ik(truth_smi) or truth_k; res = dict(id=rid, hit_A=0, hit_B=0, n_A=0, n_B=0, plan_A='', plan_B='', prank_A=-1, prank_B=-1)
    for tag, plist in parents.items():
        prods = {}
        for pr, (ps, pm) in enumerate(plist):
            tol = max(0.004, M * 10e-6)
            try:
                d = dv.derive(ps, M - pm, tol=tol, max_steps=args.steps)
            except Exception:
                d = {}
            for s, plan in d.items():
                kk = ik(s)
                if kk is None or kk in prods:
                    continue
                prods[kk] = plan
                if kk == tk and res['hit_' + tag] == 0:
                    res['hit_' + tag] = 1; res['plan_' + tag] = plan; res['prank_' + tag] = pr
        res['n_' + tag] = len(prods)
    return res


jobs, meta = [], []
for _, r in R.iterrows():
    k = r['key']
    if k not in row:
        continue
    M = mass[row[k]]
    an = [(a, v) for a, v, _ in (r.get('an_list') if isinstance(r.get('an_list'), list) else []) if v >= args.min_sim and a in row and a != k][:args.n_parents]
    orc = oracle_parents(k, args.n_parents)
    parents = dict(A=[(smi[row[a]], mass[row[a]]) for a, _ in an], B=[(smi[row[a]], mass[row[a]]) for a, _ in orc])
    jobs.append((r['id'], k, smi[row[k]], M, parents))
    meta.append(dict(id=r['id'], set=r['set'], an_top_sim=r.get('an_top_sim', 0.0), n_par_A=len(an), tanB=orc[0][1] if orc else 0.0))
t0 = time.time()
with Pool(args.workers) as pl:
    out = list(pl.imap(job, jobs, chunksize=4))
D = pd.DataFrame(meta).merge(pd.DataFrame(out), on='id')
print(f'done {len(D)} in {time.time() - t0:.0f}s')
print(D.groupby('set')[['hit_A', 'hit_B', 'n_A', 'n_B', 'n_par_A', 'tanB']].agg(['mean', 'median']).round(3).to_string())
for t in ('A', 'B'):
    print(t, 'hit plans', D.loc[D['hit_' + t] == 1, 'plan_' + t].value_counts().head(15).to_dict())
    print(t, 'hit parent rank', D.loc[D['hit_' + t] == 1, 'prank_' + t].value_counts().sort_index().to_dict())
print('hit_A by analog top sim bin', D.groupby(pd.cut(D.an_top_sim, [0, .5, .7, .8, .9, 1.01])).hit_A.agg(['mean', 'size']).round(3).to_dict())
D.to_pickle(args.out); print('GEN ORACLE DONE')
