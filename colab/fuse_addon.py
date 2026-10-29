"""Our second-opinion add-on for the public 0.477 pipeline: our own spectrum model + analog channel score the public
lists; ranks 2-25 are re-ordered by 1/(3 + public rank) + w / (3 + our rank); rank 1 is never touched.
(header below kept from gen_addon.py)

1. links our own inputs (prep cache + pools, fp_v2 checkpoint, pool popularity) into /content/work
2. runs our infer_fp.py on the (hidden) test with the analog channel and generated derivatives of the top-12 analogs
3. inserts our best generated candidate at rank --pos of the public list (rank 1 is never touched); --gate lib keeps
   molecules with a strong library match unchanged
Any failure leaves /kaggle/working/submission.csv as the public pipeline wrote it (the caller restores the control).
usage: python gen_addon.py --budget SECONDS [--pos 2] [--gate none|lib]
"""
import os, sys, glob, json, time, argparse, subprocess, shutil
import numpy as np, pandas as pd

p = argparse.ArgumentParser(); p.add_argument('--budget', type=float, required=True); p.add_argument('--pos', default='2', help='comma list of ranks for our 1st, 2nd, ... generated candidates')
p.add_argument('--w', type=float, default=0.5); p.add_argument('--sub', default='/kaggle/working/submission.csv'); p.add_argument('--gen', type=int, default=12)
args = p.parse_args(); T0 = time.time(); POS = [int(x) for x in str(args.pos).split(',')]


def ours(pat):
    h = sorted(glob.glob(f'/kaggle/input/**/satyajeetrai/**/{pat}', recursive=True), key=len)
    assert h, pat
    return h[0]


def ds(slug, f):
    h = sorted(set(glob.glob(f'/kaggle/input/**/{slug}/**/{f}', recursive=True) + glob.glob(f'/kaggle/input/**/{slug}/{f}', recursive=True)), key=len)
    assert h, (slug, f)
    return h[0]


def link(dst, src):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


comp = os.path.dirname(sorted(glob.glob('/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra/**/sample_submission.csv', recursive=True)
                              + glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True), key=len)[0])
link('/content/data/comp', comp)
for slug, f in [('coconut-casmi26-candidates', 'coco_meta.pkl'), ('casmi26-pubchemlite-cc0-candidate-fingerprints', 'pubchemlite_meta.pkl'),
                ('casmi-chebi-lmsd-structures', 'bio_meta.pkl'), ('casmi26-pubchem-tier', 'pc_mass.npy')]:
    try:
        link(f'/content/data/ds/{slug}', os.path.dirname(ds(slug, f)))
    except AssertionError as e:
        print('optional input missing', e)
link('/content/work/cache', os.path.dirname(ours('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(ours('pool.parquet')))
ck = ours('runs/fp_v2/ema.pt'); print('ours: comp', comp, '| ckpt', ck, flush=True)

out_pkl = '/kaggle/working/ours_lists.pkl'
shutil.copy2(args.sub, '/kaggle/working/public_lists.csv')
cmd = (f'cd /content/work && python -u infer_fp.py --ckpt {ck} --mode test --pubchem_top 0 --analog 100 --w_ap 0.5 --frag_top 0 --keep 100 '
       f'--cands_csv /kaggle/working/public_lists.csv --out {out_pkl}')
env = dict(os.environ, NPROC=str(max(2, (os.cpu_count() or 4))))
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=max(60, args.budget - 300), env=env)
print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
assert os.path.exists(out_pkl), 'our scoring failed'
R = pd.read_pickle(out_pkl).set_index('id')
sub = pd.read_csv(args.sub, dtype=str, keep_default_na=False)
new, n_changed, n_scored = [], 0, 0
for mid, smi in zip(sub.molecule_id, sub.smiles):
    L = smi.split(';')
    if len(L) < 3 or mid not in R.index or not isinstance(R.at[mid, 'smiles'], list) or len(R.at[mid, 'smiles']) == 0:
        new.append(smi); continue
    ours_order = list(R.at[mid, 'smiles']); n_scored += 1
    our_rank = {s: i + 1 for i, s in enumerate(ours_order)}; worst = len(ours_order) + 1
    rest = L[1:]
    sc = {s: 1.0 / (3 + j + 2) + args.w / (3 + our_rank.get(s, worst)) for j, s in enumerate(rest)}
    rest2 = sorted(rest, key=lambda s: -sc[s])
    if rest2 != rest: n_changed += 1
    new.append(';'.join([L[0]] + rest2))
out = pd.DataFrame(dict(molecule_id=sub.molecule_id, smiles=new))
assert out.molecule_id.tolist() == sub.molecule_id.tolist()
assert out.smiles.str.split(';').map(len).between(1, 25).all()
assert out.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
assert [s.split(';')[0] for s in out.smiles] == [s.split(';')[0] for s in sub.smiles]
out.to_csv(args.sub, index=False)
rep = dict(inserted=n_changed, scored=n_scored, molecules=len(out), w=args.w, seconds=round(time.time() - T0))
json.dump(rep, open('/kaggle/working/gen_addon_report.json', 'w'), indent=1); print('FUSE ADDON', rep)
