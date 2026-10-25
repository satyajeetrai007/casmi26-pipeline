"""Apply the trained LambdaRank boosters to candidate records (test_scores.pkl) and write submission.csv.
Features are built exactly as in rank_val.build (mask_frac=0). Dedupes by InChIKey14 after RDKit tautomer canonicalisation
(the metric's identity) and fills up to 25.
usage: python apply_ranker.py --scores test_scores.pkl --ranker ranker.pkl [--gl gl_test_out.pkl] --out submission.csv
"""
import argparse, pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
sys.argv_backup = list(sys.argv)
p = argparse.ArgumentParser(); p.add_argument('--scores', required=True); p.add_argument('--ranker', required=True); p.add_argument('--gl', default=''); p.add_argument('--out', required=True)
p.add_argument('--audit', default='')
args = p.parse_args()
# reuse the feature builder from rank_val by importing it with a neutral argv
sys.argv = ['rank_val.py', '--scores', args.scores] + (['--gl', args.gl] if args.gl else []) + ['--rounds', '1', '--seeds', '1']
import importlib, types
src = open('/content/work/rank_val.py', encoding='utf-8').read()
head = src.split("D = build(R, args.mask_frac)")[0]      # everything up to and including build()
ns = {}; exec(compile(head, 'rank_val_head', 'exec'), ns)
R = ns['R']; D = ns['build'](R, 0.0)
rk = pickle.load(open(args.ranker, 'rb')); feats = rk['feats']
for f in feats:
    if f not in D.columns:
        D[f] = np.nan
boosters = [lgb.Booster(model_str=s) for s in rk['boosters']]
D['score'] = np.mean([b.predict(D[feats]) for b in boosters], 0)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); canon = rdMolStandardize.TautomerEnumerator()
def metric_key(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None: return None
        return Chem.MolToInchiKey(canon.Canonicalize(m))[:14]
    except Exception:
        return None
rows = []; audit = []
for qi, r in R.iterrows():
    g = D[D.qid == qi]
    if len(g) == 0:
        rows.append(dict(molecule_id=r['id'], smiles='C')); continue
    order = np.argsort(-g.score.values, kind='stable'); smis = list(r['smiles']); keys = list(r['keys'])
    seen, out = set(), []
    for i in order:
        k = metric_key(smis[i]) or keys[i]
        if k in seen: continue
        seen.add(k); out.append(smis[i])
        if len(out) == 25: break
    rows.append(dict(molecule_id=r['id'], smiles=';'.join(out)))
    audit.append(dict(id=r['id'], n=len(g), top_src=int(np.asarray(r['src'])[order[0]]), top_score=float(g.score.values[order[0]])))
sub = pd.DataFrame(rows); sub.to_csv(args.out, index=False)
A = pd.DataFrame(audit); print('top-1 source counts (16 = PubChem-only):', A.top_src.value_counts().to_dict()); print(sub.smiles.str.count(';').describe()[['mean', 'min', 'max']])
if args.audit: A.to_csv(args.audit, index=False)
print('written', args.out, len(sub)); print('APPLY DONE')
