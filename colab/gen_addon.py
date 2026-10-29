"""Our class-3 add-on for the public 0.477 pipeline.

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
p.add_argument('--gate', default='none'); p.add_argument('--sub', default='/kaggle/working/submission.csv'); p.add_argument('--gen', type=int, default=12)
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

out_pkl = '/kaggle/working/gen_test.pkl'
cmd = (f'cd /content/work && python -u infer_fp.py --ckpt {ck} --mode test --pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5 '
       f'--gen {args.gen} --gen_pen 0 --frag_top 0 --out {out_pkl} --sub /kaggle/working/submission_ours.csv')
env = dict(os.environ, NPROC=str(max(2, (os.cpu_count() or 4))))
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=max(60, args.budget - 300), env=env)
print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
assert os.path.exists(out_pkl), 'our inference failed'

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')


def ik(s):
    try:
        m = Chem.MolFromSmiles(s)
        return None if m is None else Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None


R = pd.read_pickle(out_pkl).set_index('id')
sub = pd.read_csv(args.sub, dtype=str, keep_default_na=False)
new, n_ins, n_gated, n_nogen = [], 0, 0, 0
for mid, smi in zip(sub.molecule_id, sub.smiles):
    L = smi.split(';')
    if mid not in R.index or not isinstance(R.at[mid, 'keys'], list):
        new.append(smi); n_nogen += 1; continue
    r = R.loc[mid]
    if args.gate == 'lib' and float(r['lib_top']) >= 0.8:
        new.append(smi); n_gated += 1; continue
    g = np.asarray(r['is_gen']) > 0
    if not g.any():
        new.append(smi); n_nogen += 1; continue
    have = {ik(s) for s in L} | set(L)
    picks = []
    for i in np.where(g)[0]:                      # records are in our fused order
        s = r['smiles'][i]
        if s not in have and r['keys'][i] not in have:
            picks.append(s); have.add(s)
        if len(picks) >= len(POS):
            break
    if not picks:
        new.append(smi); continue
    for pos, s in zip(POS, picks):
        k = max(1, min(pos - 1, len(L)))
        L = L[:k] + [s] + L[k:]
    new.append(';'.join(L[:25])); n_ins += 1
out = pd.DataFrame(dict(molecule_id=sub.molecule_id, smiles=new))
assert out.molecule_id.tolist() == sub.molecule_id.tolist()
assert out.smiles.str.split(';').map(len).between(1, 25).all()
assert out.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
out.to_csv(args.sub, index=False)
rep = dict(inserted=n_ins, gated=n_gated, no_generated=n_nogen, molecules=len(out), pos=args.pos, gate=args.gate, seconds=round(time.time() - T0))
json.dump(rep, open('/kaggle/working/gen_addon_report.json', 'w'), indent=1); print('GEN ADDON', rep)
