"""Build Kaggle kernel folders (notebook + kernel-metadata.json) from the scripts in colab/.
usage: python make_kernels.py <scratchpad_dir>
"""
import json, os, sys
S = sys.argv[1]
SRC = os.path.join(S, 'colab')
COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
DS = ['prvsiyan/coconut-casmi26-candidates', 'franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints',
      'sergiomalv/casmi-chebi-lmsd-structures', 'ahmedberatozer/casmi26-pubchem-tier', 'metric/rdkit-2026-3-3-wheel']

SETUP = r'''import os, glob, subprocess, shutil, time
T0 = time.time()
def find(pat):
    h = sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True), key=len)
    assert h, pat
    return h[0]
def link(dst, src):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)
os.makedirs('/content/data/ds', exist_ok=True)
link('/content/data/comp', os.path.dirname(find('sample_submission.csv')))
for slug, f in [('coconut-casmi26-candidates', 'coco_meta.pkl'), ('casmi26-pubchemlite-cc0-candidate-fingerprints', 'pubchemlite_meta.pkl'),
                ('casmi-chebi-lmsd-structures', 'bio_meta.pkl'), ('casmi26-pubchem-tier', 'pc_mass.npy')]:
    link(f'/content/data/ds/{slug}', os.path.dirname(find(f)))
WORK = os.environ.get('CASMI_WORK', '/kaggle/working'); os.makedirs(WORK, exist_ok=True); link('/content/work', WORK)
os.chdir('/content/work'); os.environ['NPROC'] = str(max(2, os.cpu_count() - 1))
try:
    import rdkit
except ImportError:
    import sys
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    whl = sorted(glob.glob(f'/kaggle/input/**/rdkit-2026.3.3-{tag}-*.whl', recursive=True)); assert whl, 'rdkit wheel'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]])
import rdkit; print('rdkit', rdkit.__version__)
print('cpus', os.cpu_count(), 'work', WORK); print(subprocess.run('nvidia-smi -L; free -g | head -2; df -h /kaggle/working /kaggle/tmp 2>/dev/null', shell=True, capture_output=True, text=True).stdout)
'''


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}


def script_cells(names):
    out = []
    for n in names:
        s = open(os.path.join(SRC, n), encoding='utf-8').read()
        out.append(code(f'%%writefile /content/work/{n}\n' + s))
    return out


def write_kernel(folder, kid, title, cells, gpu, kernel_sources=(), dataset_sources=DS):
    os.makedirs(folder, exist_ok=True)
    json.dump(nb(cells), open(os.path.join(folder, 'kernel.ipynb'), 'w', encoding='utf-8'), indent=1)
    meta = {"id": f"satyajeetrai/{kid}", "title": title, "code_file": "kernel.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": False, "dataset_sources": list(dataset_sources),
            "competition_sources": [COMP], "kernel_sources": list(kernel_sources), "model_sources": []}
    json.dump(meta, open(os.path.join(folder, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', folder)


# 1) CPU prep kernel: cache + pools + baseline -> outputs in /kaggle/working
cpu_cells = [code(SETUP)] + script_cells(['build_train_cache.py', 'build_pools.py', 'casmi_val.py', 'val_baseline.py']) + [
    code('!cd /content/work && python -u build_train_cache.py 2>&1 | tail -8'),
    code('!cd /content/work && python -u build_pools.py 2>&1 | grep -v chunk | tail -12'),
    code('!cd /content/work && python -u val_baseline.py 2>&1 | tee val_baseline.log | tail -40'),
    code('!rm -rf /content/work/__pycache__; du -sh /kaggle/working/*; ls -la /kaggle/working/cache /kaggle/working/pools'),
]
write_kernel(os.path.join(S, 'k_cpu_prep'), 'casmi26-cpu-prep-cache-pools-baseline', 'casmi26-cpu-prep-cache-pools-baseline', cpu_cells, gpu=False)

# 2) GPU training kernel: caches from the prep kernel's output
GPU_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r'''
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
os.makedirs('/kaggle/working/runs', exist_ok=True); link('/content/work/runs', '/kaggle/working/runs')
prev = sorted(glob.glob('/kaggle/input/**/runs/*/last.pt', recursive=True))
print('previous checkpoints', prev)
'''
TRAIN_CMD = r'''import os, glob, subprocess
RUN = 'fp_v2'; init = sorted(glob.glob('/kaggle/input/**/runs/fp_v1/ema.pt', recursive=True)); prev = sorted(glob.glob(f'/kaggle/input/**/runs/{RUN}/last.pt', recursive=True))
resume = f' --resume {prev[0]}' if prev else (f' --init {init[0]}' if init else '')
cmd = f'cd /content/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_fp.py --out runs/{RUN} --epochs 5 --bs 512 --steps_per_epoch 6000 --lr 1.5e-4 --np_boost 3.0 --max_hours 4.3{resume} 2>&1 | tee -a runs/{RUN}/stdout.log'
print(cmd); os.makedirs(f'/content/work/runs/{RUN}', exist_ok=True)
subprocess.run(cmd, shell=True)
print(open(f'/content/work/runs/{RUN}/log.txt').read()[-6000:])
'''
gpu_cells = [code(GPU_SETUP)] + script_cells(['fpmodel.py', 'casmi_val.py', 'train_fp.py']) + [code(TRAIN_CMD), code('!ls -la /kaggle/working/runs/*; du -sh /kaggle/working')]
write_kernel(os.path.join(S, 'k_train_fp'), 'casmi26-train-fp', 'casmi26-train-fp', gpu_cells, gpu=True, kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp'])

# 3) inference kernel: prep outputs + trained model -> submission.csv (no internet)
INFER_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
ck = sorted(glob.glob('/kaggle/input/**/runs/*/ema.pt', recursive=True), key=len); print('checkpoints', ck)
os.environ['CKPT'] = ck[0]
"""
INFER_CMD = r"""import os, subprocess, pandas as pd, shutil
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
for extra, out in (('', 'val_scores_c1.pkl'), (' --class2', 'val_scores_c2.pkl')):
    cmd = f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode val{extra} --out /kaggle/working/{out} 2>&1 | tail -15"
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
cmd = f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode test --out /kaggle/working/test_scores.pkl --sub /kaggle/working/submission.csv 2>&1 | tail -15"
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
sub = pd.read_csv('/kaggle/working/submission.csv'); print(sub.shape, sub.smiles.str.count(';').describe()); assert sub.molecule_id.is_unique and sub.smiles.notna().all()
"""
infer_cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py']) + [code(INFER_CMD)]
write_kernel(os.path.join(S, 'k_infer_fp'), 'casmi26-infer-fp', 'casmi26-infer-fp', infer_cells, gpu=False,   # CPU: model inference on ~5k spectra is cheap
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
             dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])

# 4) debug kernel: validation-set construction only
DBG_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
"""
DBG_CMD = r"""import pandas as pd, numpy as np, sys
sys.path.insert(0, '/content/work')
from casmi_val import *
meta = load_meta(); pool = pd.read_parquet('/content/work/pools/pool.parquet', columns=['ik14', 'src_mask'])
print(pool.src_mask.value_counts()); coco = pool.loc[(pool.src_mask & 2) > 0, 'ik14'].values; print('coco', len(coco), coco[:3])
print(meta.ingest_lib.value_counts()); print('ok adducts', meta.adduct.isin(TEST_ADDUCTS).mean(), 'n_peaks>=3', (meta.n_peaks >= 3).mean())
Q = build_sets(meta, seed=0, coconut_keys=coco); print(Q.groupby('set').size()); Q.to_pickle('/kaggle/working/val_queries_seed0.pkl')
"""
write_kernel(os.path.join(S, 'k_debug_sets'), 'casmi26-debug-sets', 'casmi26-debug-sets', [code(DBG_SETUP)] + script_cells(['casmi_val.py']) + [code(DBG_CMD)], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline'])

# 5) GLACIER / ICEBERG re-scoring on the validation lists
GL_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl'); shutil.copy(find('val_scores_c2.pkl'), '/content/work/val_scores_c2.pkl'); shutil.copy(find('test_scores.pkl'), '/content/work/test_scores.pkl')
# simulator packages: copy ICEBERG and replace its cp312 RDKit wheel by the cp313 one (Kaggle image is Python 3.13)
import sys
ICE_SRC = os.path.dirname(find('ice_runner.py')); GL_SRC = os.path.dirname(find('gl_runner.py'))
ICE_PKG = '/kaggle/tmp/ice_pkg'; shutil.copytree(ICE_SRC, ICE_PKG, dirs_exist_ok=True)
W = os.path.join(ICE_PKG, 'wheels'); tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
for w in glob.glob(os.path.join(W, 'rdkit-*')):
    if tag not in w: os.remove(w)
if not glob.glob(os.path.join(W, f'rdkit-*{tag}*')):
    src = sorted(glob.glob(f'/kaggle/input/**/rdkit-2025.3.6-{tag}-*.whl', recursive=True)); assert src, 'need cp313 rdkit 2025.3.6 wheel'
    shutil.copy(src[0], os.path.join(W, os.path.basename(src[0]) + '.ice'))
print('wheels:', sorted(os.listdir(W)))
os.environ['ICE_PKG'] = ICE_PKG; os.environ['GL_PKG'] = GL_SRC; os.environ['ICE_WHEELS'] = W
"""
GL_CMD = r"""import os, subprocess
for mode, scores, out, budget in (('test', 'test_scores.pkl', 'gl_test_out.pkl', 1500), ('val', 'val_scores_c2.pkl', 'gl_val_out.pkl', 1800)):
    cmd = f"cd /content/work && python -u gl_val.py --mode {mode} --scores /content/work/{scores} --gl_pkg {os.environ['GL_PKG']} --ice_pkg {os.environ['ICE_PKG']} --wheels {os.environ['ICE_WHEELS']} --budget {budget} --top_n 40 --out /kaggle/working/{out} 2>&1 | tail -30"
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
"""
write_kernel(os.path.join(S, 'k_gl_val'), 'casmi26-gl-val', 'casmi26-gl-val', [code(GL_SETUP)] + script_cells(['gl_val.py']) + [code(GL_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-infer-fp'],
             dataset_sources=DS + ['ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'eveliaveldrine/rdkit-wheels'])

# 6) popularity join + PubChem windows (CPU)
POP_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl'); os.makedirs('/kaggle/working/pop', exist_ok=True); link('/content/work/pop', '/kaggle/working/pop')
"""
write_kernel(os.path.join(S, 'k_pop'), 'casmi26-pop', 'casmi26-pop', [code(POP_SETUP)] + script_cells(['casmi_val.py', 'build_pop.py']) + [code('!cd /content/work && python -u build_pop.py 2>&1 | tail -40')], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline'], dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])

# 7) ranker CV on validation lists (CPU)
RANK_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
for f in ('val_scores_c1.pkl', 'val_scores_c2.pkl'):
    xv = sorted(glob.glob(f'/kaggle/input/**/casmi26-xenc-val/**/{f}', recursive=True)); src = xv[0] if xv else find(f); print('using', src); shutil.copy(src, f'/content/work/{f}')
gl = sorted(glob.glob('/kaggle/input/**/gl_val_out.pkl', recursive=True)); os.environ['GLARG'] = f' --gl {gl[0]}' if gl else ''
"""
RANK_CMD = r"""import os, subprocess
cmd = f"cd /content/work && python -u rank_val.py --scores val_scores_c1.pkl,val_scores_c2.pkl{os.environ['GLARG']} --out /kaggle/working/ranker.pkl 2>&1 | tail -70"
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
"""
write_kernel(os.path.join(S, 'k_rank'), 'casmi26-rank', 'casmi26-rank', [code(RANK_SETUP)] + script_cells(['fpmodel.py', 'rank_val.py']) + [code(RANK_CMD)], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-infer-fp', 'satyajeetrai/casmi26-gl-val'] + (['satyajeetrai/casmi26-xenc-val'] if os.environ.get('XENC') else []))

# 8) submission kernel: apply the ranker to the test records (CPU, no internet)
SUB_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
shutil.copy(find('test_scores.pkl'), '/content/work/test_scores.pkl'); shutil.copy(find('ranker.pkl'), '/content/work/ranker.pkl')
gl = sorted(glob.glob('/kaggle/input/**/gl_test_out.pkl', recursive=True)); os.environ['GLARG'] = f' --gl {gl[0]}' if gl else ''; print('gl_test', gl[:1])
"""
SUB_CMD = r"""import os, subprocess, pandas as pd
cmd = f"cd /content/work && python -u apply_ranker.py --scores test_scores.pkl --ranker ranker.pkl{os.environ['GLARG']} --out /kaggle/working/submission.csv --audit /kaggle/working/audit.csv 2>&1 | tail -20"
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
sub = pd.read_csv('/kaggle/working/submission.csv'); assert sub.molecule_id.is_unique and sub.smiles.notna().all() and (sub.smiles.str.count(';') <= 24).all(); print(sub.shape)
"""
write_kernel(os.path.join(S, 'k_submit'), 'casmi26-submit', 'casmi26-submit', [code(SUB_SETUP)] + script_cells(['fpmodel.py', 'rank_val.py', 'apply_ranker.py']) + [code(SUB_CMD)], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-infer-fp', 'satyajeetrai/casmi26-rank', 'satyajeetrai/casmi26-gl-val'])

# 9) FINAL end-to-end submission kernel (GPU): test inference + simulators + ranker, all at run time on test.parquet
FINAL_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet')))
os.makedirs('/content/work/pop', exist_ok=True); shutil.copy(find('pool_pop.parquet'), '/content/work/pop/pool_pop.parquet')
ck = sorted(glob.glob('/kaggle/input/**/runs/*/ema.pt', recursive=True), key=len); print('checkpoints', ck); os.environ['CKPT'] = ck[0]
shutil.copy(find('ranker.pkl'), '/content/work/ranker.pkl')
import sys
ICE_SRC = os.path.dirname(find('ice_runner.py')); GL_SRC = os.path.dirname(find('gl_runner.py'))
ICE_PKG = '/kaggle/tmp/ice_pkg'; shutil.copytree(ICE_SRC, ICE_PKG, dirs_exist_ok=True)
W = os.path.join(ICE_PKG, 'wheels'); tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
for w in glob.glob(os.path.join(W, 'rdkit-*')):
    if tag not in w: os.remove(w)
if not glob.glob(os.path.join(W, f'rdkit-*{tag}*')):
    src = sorted(glob.glob(f'/kaggle/input/**/rdkit-2025.3.6-{tag}-*.whl', recursive=True)); assert src, 'need cp313 rdkit 2025.3.6 wheel'
    shutil.copy(src[0], os.path.join(W, os.path.basename(src[0]) + '.ice'))
os.environ['ICE_PKG'] = ICE_PKG; os.environ['GL_PKG'] = GL_SRC; os.environ['ICE_WHEELS'] = W
print('setup done', f'{time.time()-T0:.0f}s')
"""
FINAL_CMD = r"""import os, subprocess, pandas as pd, time
T1 = time.time()
def run(cmd):
    print('>>', cmd[:160]); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-6000:]); print(r.stderr[-3000:] if r.returncode else ''); print(f'[{time.time()-T1:.0f}s]')
run(f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode test --out /content/work/test_scores.pkl --sub /kaggle/working/submission_fused.csv 2>&1 | tail -8")
run(f"cd /content/work && python -u gl_val.py --mode test --scores /content/work/test_scores.pkl --gl_pkg {os.environ['GL_PKG']} --ice_pkg {os.environ['ICE_PKG']} --wheels {os.environ['ICE_WHEELS']} --budget 3000 --top_n 40 --out /content/work/gl_test_out.pkl 2>&1 | grep -v -E 'chunk:|adj = torch' | tail -12")
xk = sorted(__import__('glob').glob('/kaggle/input/**/runs/xenc_v*/ema.pt', recursive=True))
if xk:
    run(f"cd /content/work && python -u score_xenc.py --ckpt {xk[-1]} --mode test --scores /content/work/test_scores.pkl --outdir /content/work 2>&1 | tail -6")
run("cd /content/work && python -u apply_ranker.py --scores test_scores.pkl --ranker ranker.pkl --gl gl_test_out.pkl --out /kaggle/working/submission.csv --audit /kaggle/working/audit.csv 2>&1 | tail -12")
sub = pd.read_csv('/kaggle/working/submission.csv'); te = pd.read_parquet(find('test.parquet') if False else sorted(__import__('glob').glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0], columns=['molecule_id'])
assert set(sub.molecule_id) == set(te.molecule_id.unique()) and sub.molecule_id.is_unique and sub.smiles.notna().all() and (sub.smiles.str.count(';') <= 24).all(), 'submission invalid'
print('OK submission', sub.shape, f'total {time.time()-T1:.0f}s')
"""
write_kernel(os.path.join(S, 'k_final'), 'casmi26-final', 'casmi26-final',
             [code(FINAL_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'infer_fp.py', 'gl_val.py', 'rank_val.py', 'apply_ranker.py', 'xenc.py', 'molfeat.py', 'score_xenc.py']) + [code(FINAL_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop', 'satyajeetrai/casmi26-rank'] + (['satyajeetrai/casmi26-train-xenc'] if os.environ.get('XENC') else []),
             dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior', 'ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'eveliaveldrine/rdkit-wheels'])

# 10) molecule features for the cross-encoder (CPU)
MF_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/pools', os.path.dirname(find('pool.parquet'))); os.makedirs('/kaggle/working/molfeat', exist_ok=True); link('/content/work/molfeat', '/kaggle/working/molfeat')
"""
write_kernel(os.path.join(S, 'k_molfeat'), 'casmi26-molfeat', 'casmi26-molfeat', [code(MF_SETUP)] + script_cells(['molfeat.py', 'build_molfeat.py']) + [code('!cd /content/work && python -u build_molfeat.py 2>&1 | tail -8')], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline'])

# 11) cross-encoder training (GPU)
XT_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy'))); link('/content/work/pools', os.path.dirname(find('pool.parquet'))); link('/content/work/molfeat', os.path.dirname(find('mf_atoms.npy')))
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
os.makedirs('/kaggle/working/runs', exist_ok=True); link('/content/work/runs', '/kaggle/working/runs')
"""
XT_CMD = r"""import os, glob, subprocess
RUN = os.environ.get('XRUN', 'xenc_v1'); init = sorted(glob.glob('/kaggle/input/**/fp_v2_ema.pt', recursive=True) + glob.glob('/kaggle/input/**/runs/fp_v2/ema.pt', recursive=True), key=len)
prev = sorted(glob.glob(f'/kaggle/input/**/runs/{RUN}/last.pt', recursive=True)); resume = f' --resume {prev[0]}' if prev else ''
print('init', init[:1], 'resume', prev[:1]); os.makedirs(f'/content/work/runs/{RUN}', exist_ok=True)
cmd = f'cd /content/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_xenc.py --init {init[0]} --out runs/{RUN} --epochs 8 --steps 5000 --bs 32 --group 16 --max_hours 5.3{resume} 2>&1 | tee -a runs/{RUN}/stdout.log | grep -v -E "step [0-9]*[1-9]000 |step [0-9]*500 " | tail -80'
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
print(open(f'/content/work/runs/{RUN}/log.txt').read()[-5000:])
"""
write_kernel(os.path.join(S, 'k_train_xenc'), 'casmi26-train-xenc', 'casmi26-train-xenc', [code(XT_SETUP)] + script_cells(['fpmodel.py', 'xenc.py', 'train_xenc.py']) + [code(XT_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-molfeat'] + (['satyajeetrai/casmi26-train-xenc'] if os.environ.get('XENC_RESUME') else []),
             dataset_sources=DS + ['satyajeetrai/casmi26-fp-v2-ema'])

# 12) cross-encoder scoring of the validation records (GPU) -> val_scores_c1/c2.pkl with an 'xenc' column
XV_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/cache', os.path.dirname(find('peaks_mz.npy')))
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
for f in ('val_scores_c1.pkl', 'val_scores_c2.pkl'): shutil.copy(find(f), f'/content/work/{f}')
xk = sorted(glob.glob('/kaggle/input/**/runs/xenc_v*/ema.pt', recursive=True)); print('xenc', xk); os.environ['XCK'] = xk[-1]
"""
XV_CMD = r"""import os, subprocess
cmd = f"cd /content/work && python -u score_xenc.py --ckpt {os.environ['XCK']} --mode val --scores /content/work/val_scores_c1.pkl,/content/work/val_scores_c2.pkl --outdir /kaggle/working 2>&1 | tail -20"
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
"""
write_kernel(os.path.join(S, 'k_xenc_val'), 'casmi26-xenc-val', 'casmi26-xenc-val', [code(XV_SETUP)] + script_cells(['fpmodel.py', 'xenc.py', 'molfeat.py', 'score_xenc.py']) + [code(XV_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-infer-fp', 'satyajeetrai/casmi26-train-xenc'])

# 13) honest-validation inference variants (CPU, run in parallel)
def infer_variant(slug, jobs):
    cmd = "import os, subprocess, shutil\nshutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')\n"
    for extra, out in jobs:
        cmd += f"print(subprocess.run(f\"cd /content/work && python -u infer_fp.py --ckpt {{os.environ['CKPT']}} {extra} --out /kaggle/working/{out} 2>&1 | tail -12\", shell=True, capture_output=True, text=True).stdout)\n"
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py']) + [code(cmd)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
infer_variant('infer-hc-a', [('--mode val --class2 --truth_pubchem --pc_select pop', 'val_c2tp_pop.pkl')])
infer_variant('infer-hc-b', [('--mode val --class2 --truth_pubchem --pc_select fp', 'val_c2tp_fp.pkl')])
infer_variant('infer-hc-c', [('--mode val --pc_select fp', 'val_c1_fp.pkl'), ('--mode test --pc_select fp --sub /kaggle/working/submission_fused.csv', 'test_scores_fp.pkl')])

# 14) simulator scores for the honest-validation records (GPU) and ranker variants (CPU)
GLX_SETUP = GL_SETUP.replace("shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl'); shutil.copy(find('val_scores_c2.pkl'), '/content/work/val_scores_c2.pkl'); shutil.copy(find('test_scores.pkl'), '/content/work/test_scores.pkl')",
                             "shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')")
assert GLX_SETUP != GL_SETUP
def gl_variant(slug, src_kernel, jobs):
    cmd = "import os, subprocess, shutil\n"
    for mode, f, out, budget in jobs:
        cmd += f"shutil.copy(find('{f}'), '/content/work/{f}')\n"
        cmd += (f"print(subprocess.run(f\"cd /content/work && python -u gl_val.py --mode {mode} --scores /content/work/{f} --gl_pkg {{os.environ['GL_PKG']}} --ice_pkg {{os.environ['ICE_PKG']}} "
                f"--wheels {{os.environ['ICE_WHEELS']}} --budget {budget} --top_n 40 --out /kaggle/working/{out} 2>&1 | grep -v -E 'chunk:|adj = torch' | tail -16\", shell=True, capture_output=True, text=True).stdout)\n")
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, [code(GLX_SETUP)] + script_cells(['gl_val.py']) + [code(cmd)], gpu=True,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/' + src_kernel],
                 dataset_sources=DS + ['ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'eveliaveldrine/rdkit-wheels'])
gl_variant('gl-hc-a', 'casmi26-infer-hc-a', [('val', 'val_c2tp_pop.pkl', 'gl_c2tp_pop.pkl', 1800)])
gl_variant('gl-hc-bc', 'casmi26-infer-hc-b', [('val', 'val_c2tp_fp.pkl', 'gl_c2tp_fp.pkl', 1800)])
gl_variant('gl-hc-c', 'casmi26-infer-hc-c', [('val', 'val_c1_fp.pkl', 'gl_c1_fp.pkl', 1800), ('test', 'test_scores_fp.pkl', 'gl_test_fp.pkl', 1200)])

def rank_variant(slug, files, gls, extra, sources):
    setup = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + "\n"
    for f in files + gls:
        setup += f"shutil.copy(find('{f}'), '/content/work/{f}')\n"
    if '--apply' in extra:
        setup += "shutil.copy(find('ranker.pkl'), '/content/work/ranker_v1.pkl')\n"
    cmd = (f"import subprocess\nprint(subprocess.run(\"cd /content/work && python -u rank_val.py --scores {','.join(files)} {('--gl ' + ','.join(gls)) if gls else ''} {extra} --out /kaggle/working/ranker.pkl 2>&1 | tail -45\", "
           "shell=True, capture_output=True, text=True).stdout)\n")
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, [code(setup)] + script_cells(['fpmodel.py', 'rank_val.py']) + [code(cmd)], gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline'] + ['satyajeetrai/' + s for s in sources])
# calibration: the frozen V1 ranker (casmi26-rank v5 output) on V1-style class-1 records + honest class-2 records with V1's PubChem selection
rank_variant('rank-cal', ['val_scores_c1.pkl', 'val_c2tp_pop.pkl'], ['gl_val_out.pkl', 'gl_c2tp_pop.pkl'], '--apply /content/work/ranker_v1.pkl',
             ['casmi26-infer-fp', 'casmi26-gl-val', 'casmi26-infer-hc-a', 'casmi26-gl-hc-a', 'casmi26-rank'])
# new ranker: CV on class-1 + honest class-2 records with spectral-fit PubChem selection
rank_variant('rank-hc', ['val_c1_fp.pkl', 'val_c2tp_fp.pkl'], ['gl_c1_fp.pkl', 'gl_c2tp_fp.pkl'], '',
             ['casmi26-infer-hc-b', 'casmi26-infer-hc-c', 'casmi26-gl-hc-bc', 'casmi26-gl-hc-c'])

# 15) V2 submission notebook: spectral-fit PubChem selection (infer_fp default --pc_select fp) + ranker trained on honest records (rank-hc)
FINAL_HC_CMD = FINAL_CMD.replace("--mode test --out /content/work/test_scores.pkl", "--mode test --pc_select fp --out /content/work/test_scores.pkl")
assert FINAL_HC_CMD != FINAL_CMD
write_kernel(os.path.join(S, 'k_final-hc'), 'casmi26-final-hc', 'casmi26-final-hc',
             [code(FINAL_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'infer_fp.py', 'gl_val.py', 'rank_val.py', 'apply_ranker.py', 'xenc.py', 'molfeat.py', 'score_xenc.py']) + [code(FINAL_HC_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop', 'satyajeetrai/casmi26-rank-hc'],
             dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior', 'ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'eveliaveldrine/rdkit-wheels'])

# 16) natural-product-likeness of the validation structures (CPU)
NP_SETUP = SETUP.replace("WORK = os.environ.get('CASMI_WORK', '/kaggle/working')", "WORK = '/kaggle/tmp/work'") + r"""
link('/content/work/pools', os.path.dirname(find('pool.parquet'))); shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
"""
write_kernel(os.path.join(S, 'k_np-subset'), 'casmi26-np-subset', 'casmi26-np-subset', [code(NP_SETUP)] + script_cells(['fpmodel.py', 'np_subset.py']) + [code('!cd /content/work && python -u np_subset.py 2>&1 | tail -12')], gpu=False,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline'])

# 17) LB diagnostic probes (CPU submission notebooks; one component changes at a time)
def probe(slug, extra):
    cmd = ("import os, subprocess, pandas as pd, glob\n"
           f"print(subprocess.run(f\"cd /content/work && python -u infer_fp.py --ckpt {{os.environ['CKPT']}} --mode test {extra} --frag_top 0 --out /content/work/test_probe.pkl --sub /kaggle/working/submission.csv 2>&1 | tail -6\", shell=True, capture_output=True, text=True).stdout)\n"
           "sub = pd.read_csv('/kaggle/working/submission.csv'); te = pd.read_parquet(sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0], columns=['molecule_id'])\n"
           "assert set(sub.molecule_id) == set(te.molecule_id.unique()) and sub.molecule_id.is_unique and sub.smiles.notna().all() and (sub.smiles.str.count(';') <= 24).all(), 'submission invalid'\n"
           "print('OK submission', sub.shape)\n")
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py']) + [code(cmd)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
probe('probe1-pool', '--pubchem_top 0 --lib_gate 0.8')
probe('probe2-pubchem', '--pubchem_top 500 --pc_select pop --lib_gate 0.8')
probe('probe3-pop', '--pubchem_top 500 --pc_select pop --lib_gate 0.8 --pop_w 0.15')

# 18) library-token check (CPU): S1/S2/S3 pool-only fused retrieval with the np-examples token vs the enveda-180 token
infer_variant('libtok', [('--mode val --pubchem_top 0 --frag_top 0', 'v_c1_np.pkl'), ('--mode val --pubchem_top 0 --frag_top 0 --np_lib_as enveda-180', 'v_c1_e180.pkl'),
                         ('--mode val --class2 --pubchem_top 0 --frag_top 0', 'v_c2_np.pkl'), ('--mode val --class2 --pubchem_top 0 --frag_top 0 --np_lib_as enveda-180', 'v_c2_e180.pkl')])

# 19) score-decomposition probes (CPU, pool only, library gate 0.8 as in probe1): which score carries signal on the hidden test
probe('probe4-ll', '--pubchem_top 0 --lib_gate 0.8 --sub_score ll')
probe('probe5-con', '--pubchem_top 0 --lib_gate 0.8 --sub_score con')
probe('probe6-prior', '--pubchem_top 0 --lib_gate 0.8 --sub_score prior')

# 20) mass-shifted analog channel (CPU): class-2 and class-1 validation, pool only, with the analog weight sweep
infer_variant('analog-val', [('--mode val --class2 --pubchem_top 0 --frag_top 0 --analog 100', 'v_c2_analog.pkl'),
                             ('--mode val --pubchem_top 0 --frag_top 0 --analog 100', 'v_c1_analog.pkl')])

# 21) fp_v3: fp_v2 fine-tuned with the same-mass ranking loss on high-quality spectra only (GPU, 2xT4)
TRAIN_RANK_CMD = r"""import os, glob, subprocess
RUN = 'fp_v3'; init = sorted(glob.glob('/kaggle/input/**/runs/fp_v2/ema.pt', recursive=True)); print('init', init)
cmd = f'cd /content/work && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_fp.py --out runs/{RUN} --init {init[0]} --epochs 3 --bs 512 --steps_per_epoch 4000 --lr 1e-4 --np_boost 3.0 --w_rank 1.0 --hq --max_hours 2.7 2>&1 | tee -a runs/{RUN}/stdout.log'
print(cmd); os.makedirs(f'/content/work/runs/{RUN}', exist_ok=True)
subprocess.run(cmd, shell=True)
print(open(f'/content/work/runs/{RUN}/log.txt').read()[-6000:])
"""
write_kernel(os.path.join(S, 'k_train-fprank'), 'casmi26-train-fprank', 'casmi26-train-fprank',
             [code(GPU_SETUP)] + script_cells(['fpmodel.py', 'casmi_val.py', 'train_fp.py']) + [code(TRAIN_RANK_CMD), code('!ls -la /kaggle/working/runs/*; du -sh /kaggle/working')],
             gpu=True, kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp'])

# 22) analog probe: P1 + mass-shifted analog channel (w_ap 0.5) - the only change vs P1
probe('probe7-analog', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5')

# 23) pool-only ranker with the analog channel: CV on class-1 + class-2 records (no PubChem, no simulators)
rank_variant('rank-analog', ['v_c1_analog.pkl', 'v_c2_analog.pkl'], [], '', ['casmi26-analog-val'])

# 24) train-demotion probe: P7 + training structures after all others when the best library match < 0.8
probe('probe8-demote', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5 --demote_train 0.8')

# 25) P9 = P7 (pool only, analog w 0.5, lib gate 0.8) + GLACIER/ICEBERG same-formula re-scoring of the top 40 (fixed weights, no ranker)
P9_CMD = FINAL_CMD.split("run(f\"cd /content/work && python -u infer_fp.py")[0] + r"""
run(f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode test --pubchem_top 0 --frag_top 0 --analog 100 --w_ap 0.5 --lib_gate 0.8 --out /content/work/test_scores.pkl --sub /kaggle/working/submission_p7.csv 2>&1 | tail -8")
run(f"cd /content/work && python -u gl_val.py --mode test --scores /content/work/test_scores.pkl --gl_pkg {os.environ['GL_PKG']} --ice_pkg {os.environ['ICE_PKG']} --wheels {os.environ['ICE_WHEELS']} --budget 3000 --top_n 40 --out /content/work/gl_test_out.pkl 2>&1 | grep -v -E 'chunk:|adj = torch' | tail -12")
import shutil
gl_ok = os.path.exists('/content/work/gl_test_out.pkl')
run("cd /content/work && python -u fixed_rerank.py --scores test_scores.pkl " + ("--gl gl_test_out.pkl " if gl_ok else "") + "--lam 1.0 --top_n 40 --lib_gate 0.8 --out /kaggle/working/submission.csv 2>&1 | tail -8")
if not os.path.exists('/kaggle/working/submission.csv'): shutil.copy('/kaggle/working/submission_p7.csv', '/kaggle/working/submission.csv'); print('FALLBACK to P7 order')
sub = pd.read_csv('/kaggle/working/submission.csv'); te = pd.read_parquet(sorted(__import__('glob').glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0], columns=['molecule_id'])
assert set(sub.molecule_id) == set(te.molecule_id.unique()) and sub.molecule_id.is_unique and sub.smiles.notna().all() and (sub.smiles.str.count(';') <= 24).all(), 'submission invalid'
print('OK submission', sub.shape, f'total {time.time()-T1:.0f}s')
"""
assert 'def run(cmd)' in P9_CMD
write_kernel(os.path.join(S, 'k_final-p9'), 'casmi26-final-p9', 'casmi26-final-p9',
             [code(FINAL_SETUP.replace("shutil.copy(find('ranker.pkl'), '/content/work/ranker.pkl')", 'pass'))] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py', 'gl_val.py', 'fixed_rerank.py']) + [code(P9_CMD)], gpu=True,
             kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
             dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior', 'ahmedberatozer/casmi26-iceberg', 'ahmedberatozer/casmi26-glacier', 'eveliaveldrine/rdkit-wheels'])

# 26) analog variants + fp_v3 model (CPU, class-2 pool only): power 2, wide window, fp_v3 checkpoint
def infer_variant2(slug, jobs, extra_sources=()):
    cmd = "import os, subprocess, shutil, glob\nshutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')\nCK = {r: sorted(glob.glob(f'/kaggle/input/**/runs/{r}/ema.pt', recursive=True)) for r in ('fp_v2', 'fp_v3')}; print(CK)\n"
    for run_, extra, out in jobs:
        cmd += f"print(subprocess.run(f\"cd /content/work && python -u infer_fp.py --ckpt {{CK['{run_}'][0]}} {extra} --out /kaggle/working/{out} 2>&1 | grep -A 30 'ANALOG SWEEP' | head -24\", shell=True, capture_output=True, text=True).stdout)\n"
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py']) + [code(cmd)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'] + list(extra_sources),
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
B = '--mode val --class2 --pubchem_top 0 --frag_top 0'
infer_variant2('analog-var', [('fp_v2', B + ' --analog 100 --ap_power 2', 'v_c2_pow2.pkl'), ('fp_v2', B + ' --analog 200 --analog_win 300', 'v_c2_wide.pkl'),
                              ('fp_v3', B + ' --analog 100', 'v_c2_fpv3.pkl')], ['satyajeetrai/casmi26-train-fprank'])


# 27) class-3 feasibility: derivatives of spectral analogs / Tanimoto-oracle parents (CPU)
GEN_CMD = r'''import os, subprocess, shutil, glob
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
ck = sorted(glob.glob('/kaggle/input/**/runs/fp_v2/ema.pt', recursive=True))[0]
prev = glob.glob('/kaggle/input/**/v_c2_anl.pkl', recursive=True)
if prev: shutil.copy(prev[0], '/kaggle/working/v_c2_anl.pkl')
else: print(subprocess.run(f"cd /content/work && python -u infer_fp.py --ckpt {ck} --mode val --class2 --pubchem_top 0 --frag_top 0 --analog 100 --out /kaggle/working/v_c2_anl.pkl 2>&1 | tail -5", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run('cd /content/work && python -u gen_oracle.py --records /kaggle/working/v_c2_anl.pkl --out /kaggle/working/gen_oracle.pkl 2>&1 | tail -40', shell=True, capture_output=True, text=True).stdout)
'''
def gen_oracle_kernel(slug):
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'infer_fp.py', 'derive.py', 'gen_oracle.py']) + [code(GEN_CMD)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop', 'satyajeetrai/casmi26-gen-oracle'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
gen_oracle_kernel('gen-oracle')

# 28) class-3 channel validation: generated derivatives in the candidate list, class-2 (truth in pool) and class-3 (truth removed) sims
GENV_CMD = r'''import os, subprocess, shutil, glob
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
ck = sorted(glob.glob('/kaggle/input/**/runs/fp_v2/ema.pt', recursive=True))[0]
B = '--mode val --class2 --pubchem_top 0 --frag_top 0 --analog 100 --w_ap 0.5 --gen 6 --gen_pen 0'
for extra, out in (('', 'v_c2_gen.pkl'), ('--truth_pubchem', 'v_c3_gen.pkl')):
    print('=====', extra or 'class-2 (truth in pool)')
    print(subprocess.run(f"cd /content/work && python -u infer_fp.py --ckpt {ck} {B} {extra} --out /kaggle/working/{out} 2>&1 | grep -v '^  [0-9]' | tail -45", shell=True, capture_output=True, text=True).stdout)
'''
def genval_kernel(slug):
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py']) + [code(GENV_CMD)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
genval_kernel('gen-val')

# 29) P10 probe: P7 + class-3 derivative channel (6 parents, no penalty)
def probe_gen(slug, extra):
    cmd = ("import os, subprocess, pandas as pd, glob, numpy as np\n"
           f"print(subprocess.run(f\"cd /content/work && python -u infer_fp.py --ckpt {{os.environ['CKPT']}} --mode test {extra} --frag_top 0 --out /kaggle/working/test_probe.pkl --sub /kaggle/working/submission.csv 2>&1 | tail -6\", shell=True, capture_output=True, text=True).stdout)\n"
           "sub = pd.read_csv('/kaggle/working/submission.csv'); te = pd.read_parquet(sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0], columns=['molecule_id'])\n"
           "assert set(sub.molecule_id) == set(te.molecule_id.unique()) and sub.molecule_id.is_unique and sub.smiles.notna().all() and (sub.smiles.str.count(';') <= 24).all(), 'submission invalid'\n"
           "R = pd.read_pickle('/kaggle/working/test_probe.pkl'); g = R.is_gen.apply(lambda a: np.asarray(a)[:1].sum() if isinstance(a, np.ndarray) else 0)\n"
           "print('n_gen median', R.n_gen.median(), '| top-1 generated', int(g.sum()), 'of', len(R), '| generated in top 5', float(R.is_gen.apply(lambda a: np.asarray(a)[:5].sum() if isinstance(a, np.ndarray) else 0).mean()))\n"
           "print('OK submission', sub.shape)\n")
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py']) + [code(cmd)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
probe_gen('probe10-gen', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5 --gen 6 --gen_pen 0')
GENV12 = GENV_CMD.replace('--gen 6 --gen_pen 0', '--gen 12 --gen_pen 0')
def genval12_kernel(slug):
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py']) + [code(GENV12)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
genval12_kernel('gen-val12')
probe_gen('probe11-gen12', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5 --gen 12 --gen_pen 0')

# 30) generator variants on saved records (CPU, fast): edit set, parents, steps
GENX_CMD = r'''import os, subprocess, shutil, glob
prev = glob.glob('/kaggle/input/**/v_c2_anl.pkl', recursive=True); shutil.copy(prev[0], '/kaggle/working/v_c2_anl.pkl')
for mod, npar, steps in (('derive', 12, 2), ('derive2', 12, 2), ('derive2', 20, 2), ('derive2', 12, 3)):
    print('=====', mod, 'parents', npar, 'steps', steps, flush=True)
    print(subprocess.run(f'cd /content/work && timeout 5400 python -u gen_oracle.py --records /kaggle/working/v_c2_anl.pkl --module {mod} --n_parents {npar} --steps {steps} --skip_oracle --out /kaggle/working/genx_{mod}_{npar}_{steps}.pkl 2>&1 | grep -v Warning | tail -12', shell=True, capture_output=True, text=True).stdout, flush=True)
'''
def genx_kernel(slug):
    cells = [code(INFER_SETUP)] + script_cells(['derive.py', 'derive2.py', 'gen_oracle.py']) + [code(GENX_CMD)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-gen-oracle'],
                 dataset_sources=DS)
genx_kernel('gen-x')
GENF = GENV_CMD.replace('--gen 6 --gen_pen 0', '--gen 12 --gen_pen 0 --frag_gen 40').replace("(('', 'v_c2_gen.pkl'), ('--truth_pubchem', 'v_c3_gen.pkl'))", "(('--truth_pubchem', 'v_c3_genf.pkl'), ('', 'v_c2_genf.pkl'))")
def genf_kernel(slug):
    cells = [code(INFER_SETUP)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py']) + [code(GENF)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
genf_kernel('gen-frag')
probe_gen('probe12-genins2', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5 --gen 12 --gen_pen 0 --gen_insert 2')

# 31) test of the second-opinion add-on on a 400-row list file (CPU)
FUSET_CMD = r'''import os, glob, shutil, subprocess
os.makedirs('/kaggle/tmp/work', exist_ok=True)
src = [p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'probe9' in p or 'final-p9' in p or 'probe10' in p][0]; print('lists from', src)
shutil.copy(src, '/kaggle/working/submission.csv')
r = subprocess.run('cd /content/work && python -u fuse_addon.py --budget 5000 --w 0.5', shell=True, capture_output=True, text=True); print(r.stdout[-4000:]); print(r.stderr[-3000:])
import pandas as pd
a = pd.read_csv('/kaggle/working/public_lists.csv', dtype=str, keep_default_na=False); b = pd.read_csv('/kaggle/working/submission.csv', dtype=str, keep_default_na=False)
print('lists changed', int((a.smiles != b.smiles).sum()), 'leaders changed', int((a.smiles.str.split(';').str[0] != b.smiles.str.split(';').str[0]).sum()))
'''
def fuset_kernel(slug):
    cells = [code("import os\nos.makedirs('/kaggle/tmp/work', exist_ok=True)\nos.makedirs('/content', exist_ok=True)\nos.path.exists('/content/work') or os.symlink('/kaggle/tmp/work', '/content/work')\nimport subprocess, sys, glob\ntry:\n    import rdkit\nexcept ImportError:\n    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'\n    whl = sorted(glob.glob(f'/kaggle/input/**/rdkit-2026.3.3-{tag}-*.whl', recursive=True)); subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]])\n")] + \
            script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py', 'gen_addon.py', 'fuse_addon.py']) + [code(FUSET_CMD)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop', 'satyajeetrai/casmi26-probe10-gen'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
fuset_kernel('fuse-test')

# 32) hard400 fixture (400 held-out training molecules, test-like masking): our pipeline configs scored on it (CPU)
H400_PACKED = open(os.path.join(S, 'h400_packed.txt')).read().strip()
H400_UNPACK = "import base64, io, tarfile, os\nos.makedirs('/kaggle/tmp/h400fix', exist_ok=True)\n" + \
    "tarfile.open(fileobj=io.BytesIO(base64.b64decode('" + H400_PACKED + "')), mode='r:*').extractall('/kaggle/tmp/h400fix', filter='data')\nprint(os.listdir('/kaggle/tmp/h400fix'))\n"
H400_CMD = r'''import os, subprocess, shutil, time
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
def run(cmd):
    t = time.time(); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-2500:], r.stderr[-2500:], f'[{time.time()-t:.0f}s]', flush=True)
base = f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode test --h400 /kaggle/tmp/h400fix --frag_top 0"
JOBS = H400_JOBS
for name, extra in JOBS:
    run(f"{base} {extra} --out /kaggle/working/h400_{name}.pkl --sub /kaggle/working/h400_{name}.csv")
run('cd /content/work && python -u h400_eval.py --fix /kaggle/tmp/h400fix --s1 /content/work/val_queries_seed0.pkl --out /kaggle/working/h400_ranks.csv ' + ' '.join(f'{n}=/kaggle/working/h400_{n}.csv' for n, _ in JOBS))
'''
def h400_kernel(slug, jobs, extra_sources=()):
    cells = [code(INFER_SETUP), code(H400_UNPACK)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py', 'h400_eval.py']) + \
            [code(H400_CMD.replace('H400_JOBS', repr(jobs)))]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'] + list(extra_sources),
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
h400_kernel('h400-ours', [('p7', '--pubchem_top 0 --lib_gate 0.8 --analog 100 --w_ap 0.5'), ('p1', '--pubchem_top 0 --lib_gate 0.8')])

# 33) hard400: our model + analog channel score the public pipeline's own lists on the fixture (needs casmi26-h400-base output)
H400C_CMD = r'''import os, glob, shutil, subprocess, tarfile, time
import pandas as pd
shutil.copy(find('val_queries_seed0.pkl'), '/content/work/val_queries_seed0.pkl')
base = [p for p in glob.glob('/kaggle/input/**/submission.csv', recursive=True) if 'h400-base' in p]; print('base outputs', base)
os.makedirs('/kaggle/working/lists', exist_ok=True)
shutil.copy(base[0], '/kaggle/working/lists/final.csv')
for f in glob.glob(os.path.dirname(base[0]) + '/*'):
    if f.endswith('.csv') or f.endswith('.json'): shutil.copy(f, '/kaggle/working/lists/' + os.path.basename(f))
tg = glob.glob(os.path.dirname(base[0]) + '/diagnostic-details.tar.gz')
if tg:
    t = tarfile.open(tg[0]); names = [n for n in t.getnames() if n.endswith('.csv') and 'submission' in os.path.basename(n)]; print('lists in archive', names)
    for n in names:
        t.extract(n, '/kaggle/tmp/lists', filter='data'); shutil.copy('/kaggle/tmp/lists/' + n, '/kaggle/working/lists/' + n.replace('/', '__'))
L = sorted(glob.glob('/kaggle/working/lists/*.csv')); L = [p for p in L if 'final' in p] + [p for p in L if 'final' not in p]
U = {}
for p in L:
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    if list(d.columns[:2]) != ['molecule_id', 'smiles']: continue
    for m, s in zip(d.molecule_id, d.smiles):
        u = U.setdefault(m, [])
        for x in s.split(';'):
            if x and x not in u: u.append(x)
pd.DataFrame(dict(molecule_id=list(U), smiles=[';'.join(v) for v in U.values()])).to_csv('/kaggle/working/union.csv', index=False)
print('union lists', len(U), 'median size', pd.Series([len(v) for v in U.values()]).median(), flush=True)
def run(cmd):
    t = time.time(); r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print(r.stdout[-3000:], r.stderr[-2500:], f'[{time.time()-t:.0f}s]', flush=True)
run(f"cd /content/work && python -u infer_fp.py --ckpt {os.environ['CKPT']} --mode test --h400 /kaggle/tmp/h400fix --frag_top 0 --pubchem_top 0 --analog 100 --w_ap 0.5 --keep 400 --cands_csv /kaggle/working/union.csv --out /kaggle/working/h400_cands.pkl")
run('cd /content/work && python -u h400_eval.py --fix /kaggle/tmp/h400fix --s1 /content/work/val_queries_seed0.pkl --out /kaggle/working/h400_pub_ranks.csv ' + ' '.join(f'{os.path.basename(p)[:-4]}={p}' for p in L if pd.read_csv(p, nrows=1).columns[:2].tolist() == ['molecule_id', 'smiles']))
'''
def h400c_kernel(slug):
    cells = [code(INFER_SETUP), code(H400_UNPACK)] + script_cells(['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py', 'h400_eval.py']) + [code(H400C_CMD)]
    write_kernel(os.path.join(S, 'k_' + slug), 'casmi26-' + slug, 'casmi26-' + slug, cells, gpu=False,
                 kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop', 'satyajeetrai/casmi26-h400-base'],
                 dataset_sources=DS + ['dmitriigluzdov/casmi26-pubchem-popularity-prior'])
h400c_kernel('h400-cands')
