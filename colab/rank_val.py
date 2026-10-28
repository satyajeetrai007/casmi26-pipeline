"""LambdaRank re-ranker over the candidate records written by infer_fp.py (validation molecules, 5-fold CV by molecule).
Simulates PubChem-only answers: for a random share of queries the truth's pool membership is masked (src -> PubChem only),
so the ranker learns to trust PubChem-window candidates. Prints MRR@25 per set before/after and saves the boosters.
usage: python rank_val.py --scores val_scores.pkl [--gl gl_val_out.pkl] --out ranker.pkl
"""
import argparse, pickle, numpy as np, pandas as pd, lightgbm as lgb
from fpmodel import formula_elements
p = argparse.ArgumentParser(); p.add_argument('--scores', required=True); p.add_argument('--gl', default=''); p.add_argument('--out', default='ranker.pkl')
p.add_argument('--seeds', type=int, default=3); p.add_argument('--mask_frac', type=float, default=0.0); p.add_argument('--rounds', type=int, default=500)
p.add_argument('--apply', default='', help='evaluate a frozen ranker.pkl on these records instead of cross-validating')
args = p.parse_args()
FILES = args.scores.split(','); GLF = args.gl.split(',') if args.gl else []
assert len(GLF) in (0, 1, len(FILES)), 'pass one --gl per --scores file (or a single one)'
parts = []
for i_, f_ in enumerate(FILES):
    Ri = pd.read_pickle(f_); Ri['_gl'] = (i_ if len(GLF) == len(FILES) else 0) if GLF else -1; parts.append(Ri)
R = pd.concat(parts, ignore_index=True)
if 'class2' in R.columns:
    R['set'] = R['set'].astype(str) + np.where(R['class2'].fillna(False).astype(bool), '|c2', '|c1')
GLS = [pd.read_pickle(g) for g in GLF]
GL = GLS[0] if GLS else None
rng = np.random.default_rng(0)
POPCOLS = [c[4:] for c in R.columns if c.startswith('pop_')]


def z(x):
    x = np.asarray(x, np.float64); return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else x * 0


def build(R, mask_frac):
    rows = []
    for qi, r in R.iterrows():
        keys = np.asarray(r.get('keys', [])); n = len(keys)
        if n == 0: continue
        src = np.asarray(r['src'], np.int16).copy(); label = (keys == r['key']).astype(int)
        masked = 0
        if mask_frac > 0 and label.any() and rng.random() < mask_frac:
            t = int(np.argmax(label))
            if 'pop_lsid' in r and (r['pop_lsid'][t] > 0 or r['pop_lpmid'][t] > 0):   # truth documented in PubChem -> pretend PubChem-only
                src[t] = 16; masked = 1
        ll = np.asarray(r['ll']); con = np.asarray(r['con']); libsc = np.asarray(r['libsc']); forms = np.asarray(r['formula'])
        fc = pd.Series(forms).map(pd.Series(forms).value_counts()).values
        f = pd.DataFrame({'qid': qi, 'set': r['set'], 'label': label, 'masked': masked,
                          'll': ll, 'con': con, 'z_ll': z(ll), 'z_con': z(con), 'r_ll': np.argsort(np.argsort(-ll)), 'r_con': np.argsort(np.argsort(-con)),
                          'll_gap': ll.max() - ll, 'con_gap': con.max() - con, 'n_cands': n, 'n_same_formula': fc, 'ppm': (np.asarray(r['mass']) - r['mass']) / r['mass'] * 1e6,
                          'libsc': libsc, 'lib_top': float(r['lib_top']), 'libsc_is_top': ((libsc >= float(r['lib_top']) - 1e-6) & (libsc > 0)).astype(int),
                          'analog': np.asarray(r['analog']), 'analog_top': np.asarray(r['analog_top']), 'n_hits': int(r.get('n_hits', 0)),
                          'truth_pop': 0.0,
                          'n_pubchem': int(r.get('n_pubchem', 0)), 'n_spec': int(r.get('n_spec', 1))})
        for c in POPCOLS:
            if c in ('src_mask', 'np'):
                continue                      # decomposed below without the leaking bits
            f['pop_' + c] = np.asarray(r['pop_' + c], np.float32)
        if 'pop_src_mask' in r:               # PubChem depositor groups; bit 1 (spectral library) is dropped: class-2 test molecules never have it
            sm = np.asarray(r['pop_src_mask'], np.float32).astype(np.int64)
            for bit, name in ((2, 'metab_db'), (4, 'np_db'), (8, 'chebi'), (16, 'lipidmaps'), (32, 'drug_db'), (128, 'tox'), (256, 'vendor'), (512, 'wikidata')):
                f['dep_' + name] = ((sm & bit) > 0).astype(np.float32)
        if 'pop_np' in r:                     # natural-product membership; bit 1 (COCONUT) comes from the train-excluding public pool -> dropped
            npb = np.asarray(r['pop_np'], np.float32).astype(np.int64)
            f['np_lotus'] = ((npb & 2) > 0).astype(np.float32); f['np_npatlas'] = ((npb & 4) > 0).astype(np.float32)
        if label.any() and 'pop_lsid' in f:
            t = int(np.argmax(label)); f['truth_pop'] = float(r['pop_lsid'][t] + r['pop_lpmid'][t])
        if 'pop_lsid' in f:
            f['z_pop'] = z(f.pop_lsid.values + f.pop_lpmid.values); f['pop_rank'] = np.argsort(np.argsort(-(f.pop_lsid.values + f.pop_lpmid.values)))
        for extra in ('ap', 'ap1', 'ap_best_tan', 'ap_top_tan', 'ap_mean_tan'):     # mass-shifted analog channel (analog.py)
            if extra in r and isinstance(r[extra], np.ndarray) and len(r[extra]) == n:
                f[extra] = np.asarray(r[extra], np.float32)
        if 'ap' in f:
            f['ap_z'] = z(f.ap.values); f['ap_gap'] = f.ap.values.max() - f.ap.values; f['r_ap'] = np.argsort(np.argsort(-f.ap.values))
            f['an_top_sim'] = float(r.get('an_top_sim', 0.0)); f['ap_x_ll'] = f.ap_z.values * f.z_ll.values
        for extra in ('frag_w', 'frag_e', 'frag_p', 'xenc'):
            if extra in r and r[extra] is not None and np.ndim(r[extra]) == 1:
                f[extra] = np.asarray(r[extra], np.float32)
        if 'xenc' in f:
            v = f.xenc.values.astype(np.float64); fin = np.isfinite(v); vv = np.where(fin, v, np.nanmin(v) if fin.any() else 0.0)
            f['xenc_z'] = z(vv); f['xenc_gap'] = vv.max() - vv; f['r_xenc'] = np.argsort(np.argsort(-vv))
        if 'frag_w' in f:
            v = f.frag_w.values; f['frag_w_z'] = z(np.nan_to_num(v, nan=np.nanmin(v) if np.isfinite(v).any() else 0.0)); f['frag_gap'] = (np.nanmax(v) - v) if np.isfinite(v).any() else np.nan
        if r.get('el_pred') is not None and np.ndim(r['el_pred']) == 1:
            E = np.stack([formula_elements(x) for x in forms]); d = E - np.asarray(r['el_pred'])[None, :]
            f['el_l1'] = np.abs(d).sum(1); f['el_l1_z'] = z(f.el_l1.values); f['el_c_err'] = d[:, 0]; f['el_n_err'] = d[:, 2]; f['el_o_err'] = d[:, 3]
        GLr = GLS[int(r['_gl'])] if GLS and int(r['_gl']) >= 0 else None
        if GLr is not None:
            smis = list(r['smiles'])
            for name in ('GL', 'ICE'):
                d = GLr.get(name, {}).get(r['id'], {}); v = np.array([d.get(s, np.nan) if d.get(s) is not None else np.nan for s in smis], np.float32)
                f[name.lower()] = v; f[name.lower() + '_z'] = z(np.nan_to_num(v, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0)); f[name.lower() + '_has'] = np.isfinite(v).astype(int)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


D = build(R, args.mask_frac); feats = [c for c in D.columns if c not in ('qid', 'set', 'label', 'masked', 'truth_pop', '_fold')]
PRIOR_FEATS = [c for c in feats if c.startswith('pop_') or c.startswith('dep_') or c.startswith('np_') or c in ('z_pop', 'n_cands', 'n_same_formula', 'n_pubchem', 'n_spec', 'ppm')]
print('rows', len(D), 'queries', D.qid.nunique(), 'positives', int(D.label.sum()), 'masked queries', int(D.groupby('qid').masked.first().sum()), 'features', len(feats))


def mrr_of(df, score, by='set'):
    out = {}
    for s, g in df.assign(score=score).groupby(by):
        s = '|'.join(map(str, s)) if isinstance(s, tuple) else s
        rr = []
        for _, q in g.groupby('qid'):
            order = np.argsort(-q.score.values, kind='stable')[:25]; lab = q.label.values[order]; pos = np.where(lab == 1)[0]
            rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
        out[s] = round(float(np.mean(rr)), 4)
    return out


print('baseline fused (ll+con) ', mrr_of(D, D.z_ll + D.z_con)); print('baseline ll            ', mrr_of(D, D.ll)); print('baseline con           ', mrr_of(D, D.con))
if 'z_pop' in D: print('fused + 0.3 z_pop       ', mrr_of(D, D.z_ll + D.z_con + 0.3 * D.z_pop))
def save_rr(D, pred, path):
    rows = []
    for qid, g in D.assign(score=pred).groupby('qid'):
        order = np.argsort(-g.score.values, kind='stable')[:25]; lab = g.label.values[order]; pos = np.where(lab == 1)[0]
        rows.append(dict(qid=qid, id=R.loc[qid, 'id'], set=g.set.iloc[0], key=R.loc[qid, 'key'], rr=(1.0 / (pos[0] + 1)) if len(pos) else 0.0))
    pd.DataFrame(rows).to_csv(path, index=False); print('saved per-query rr ->', path)


def lb_estimate(D, pred):
    m = mrr_of(D, pred); tp = D.groupby('qid').truth_pop.first(); pos = tp[tp > 0]
    terc = pd.qcut(pos, 3, labels=['pop_low', 'pop_mid', 'pop_high']) if len(pos) >= 3 else pd.Series(dtype=object)
    D = D.assign(pop_terc=np.where(D.qid.map(tp) > 0, D.qid.map(terc).astype(str), 'pop_zero')); byt = mrr_of(D, pred, by=['set', 'pop_terc'])
    c1 = m.get('S1|c1', np.nan); c2_all = m.get('S3|c2', np.nan); c2_lm = np.nanmean([byt.get('S3|c2|pop_low', np.nan), byt.get('S3|c2|pop_mid', np.nan)])
    print('MRR by set:', m); print('by set x truth-popularity:', byt)
    print(f'LB estimate A (0.16*S1|c1 + 0.55*S3|c2 low/mid pop): {0.16 * np.nan_to_num(c1) + 0.55 * c2_lm:.4f} | B (0.16*S1|c1 + 0.55*S3|c2 all): {0.16 * np.nan_to_num(c1) + 0.55 * c2_all:.4f}')


if args.apply:
    rk = pickle.load(open(args.apply, 'rb'))
    for f_ in rk['feats']:
        if f_ not in D.columns: D[f_] = np.nan
    pred_a = np.mean([lgb.Booster(model_str=s_).predict(D[rk['feats']]) for s_ in rk['boosters']], 0)
    print('=== frozen ranker', args.apply, 'on', args.scores); lb_estimate(D, pred_a); save_rr(D, pred_a, '/kaggle/working/rr_apply.csv')
    print('=== fused (ll+con) on the same records'); lb_estimate(D, (D.z_ll + D.z_con).values)
    print('RANK DONE'); raise SystemExit(0)
# folds by STRUCTURE, not by record: the class-1 and class-2 files hold the same molecules with near-identical candidate lists,
# so a record-level split puts the same molecule on both sides and the ranker memorises it (inflated every earlier CV).
qkey = D.qid.map(R['key'].astype(str)); ukeys = qkey.unique(); rng.shuffle(ukeys); kfold = {k: i % 5 for i, k in enumerate(ukeys)}
D['_fold'] = qkey.map(kfold).values; folds = [D.qid[D._fold == i].unique() for i in range(5)]
print('folds by structure:', [len(f) for f in folds], 'records;', len(ukeys), 'structures')
pred = np.zeros(len(D)); boosters = []
params = dict(objective='lambdarank', metric='ndcg', eval_at=[25], learning_rate=0.03, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, lambdarank_truncation_level=30)
for fi, fq in enumerate(folds):
    te = D.qid.isin(fq).values; tr = ~te
    Dtr = D[tr].sort_values('qid', kind='stable'); grp_tr = Dtr.groupby('qid', sort=False).size().values
    for seed in range(args.seeds):
        bst = lgb.train({**params, 'seed': seed}, lgb.Dataset(Dtr[feats], Dtr.label, group=grp_tr), num_boost_round=args.rounds)
        pred[te] += bst.predict(D.loc[te, feats]) / args.seeds; boosters.append(bst)
    print(f'fold {fi} done', flush=True)
print('ranker (5-fold CV)      ', mrr_of(D, pred)); lb_estimate(D, pred); save_rr(D, pred, '/kaggle/working/rr_cv.csv')
# honest class-2 view: masked queries (truth treated as PubChem-only) split by how documented the truth is in PubChem
tp = D.groupby('qid').truth_pop.first(); terc = pd.qcut(tp[tp > 0], 3, labels=['pop_low', 'pop_mid', 'pop_high'])
D['pop_terc'] = D.qid.map(terc).astype(str); D['pop_terc'] = np.where(D.qid.map(tp) > 0, D.pop_terc, 'pop_zero')
print('ranker by set x truth-popularity tercile:', mrr_of(D, pred, by=['set', 'pop_terc']))
est = {}
for mode in ('c1', 'c2'):
    s1 = mrr_of(D, pred).get(f'S1|{mode}', np.nan); byt = mrr_of(D, pred, by=['set', 'pop_terc'])
    x = np.nanmean([byt.get(f'S3|c2|pop_low', np.nan), byt.get(f'S3|c2|pop_mid', np.nan)])
    est[mode] = 0.16 * s1 + 0.55 * x
print('LB estimate (0.16*class1[S1|c1] + 0.55*class2[S3|c2 low/mid popularity]):', round(float(est['c1']), 4), '| class2 proxy X =', round(float(np.nanmean([mrr_of(D, pred, by=['set','pop_terc']).get('S3|c2|pop_low', np.nan), mrr_of(D, pred, by=['set','pop_terc']).get('S3|c2|pop_mid', np.nan)])), 4))
# spectrum-blind check: priors-only ranker
pred_p = np.zeros(len(D))
for fi, fq in enumerate(folds):
    te = D.qid.isin(fq).values; tr = ~te; Dtr = D[tr].sort_values('qid', kind='stable'); grp_tr = Dtr.groupby('qid', sort=False).size().values
    bst = lgb.train({**params, 'seed': 0}, lgb.Dataset(Dtr[PRIOR_FEATS], Dtr.label, group=grp_tr), num_boost_round=200); pred_p[te] = bst.predict(D.loc[te, PRIOR_FEATS])
print('priors-only ranker (spectrum-blind):', mrr_of(D, pred_p))
D['masked_set'] = D.set + np.where(D.masked == 1, '|pc-only', '|pool'); print('ranker by truth source  ', mrr_of(D, pred, by='masked_set'))
imp = pd.Series(np.mean([b.feature_importance('gain') for b in boosters], 0), index=feats).sort_values(ascending=False); print(imp.head(20).round(0))
pickle.dump(dict(feats=feats, boosters=[b.model_to_string() for b in boosters], params=params), open(args.out, 'wb'))
print('RANK DONE')
