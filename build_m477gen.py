import json, glob, os, sys
S = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(S, 'colab')
POS = sys.argv[1] if len(sys.argv) > 1 else '2'; GATE = sys.argv[2] if len(sys.argv) > 2 else 'none'; SLUG = sys.argv[3] if len(sys.argv) > 3 else 'casmi26-m477-gen'; ADDON = sys.argv[4] if len(sys.argv) > 4 else 'gen'; FAST = len(sys.argv) > 5 and sys.argv[5] == 'fast'
ADDON_ARGS = f'--pos {POS} --gate {GATE}' if ADDON == 'gen' else f'--w {GATE}'
base = json.load(open(glob.glob(os.path.join(S, 'pub/m477/*.ipynb'))[0], encoding='utf8'))
meta = json.load(open(os.path.join(S, 'pub/m477/kernel-metadata.json'), encoding='utf8'))
def code(src): return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
cells = [dict(c, outputs=[], execution_count=None) if c['cell_type'] == 'code' else c for c in base['cells']]
if FAST:
    src = ''.join(cells[1]['source']); assert "COMMIT_MODE = 'smoke'" in src; cells[1]['source'] = src.replace("COMMIT_MODE = 'smoke'", "COMMIT_MODE = 'fast'")
cells.insert(0, md("# Fork of busyaprime's LB 0.477 notebook + our class-3 derivative channel\n\n"
                   "Everything above our add-on is the public 0.477 notebook unchanged (Apache-2.0 code; credits in its section 8). "
                   f"Our add-on (last cells): our own fingerprint model, mass-shifted analog search and a derivative generator "
                   f"(methyl / hydroxyl / sugar / acyl / prenyl edits of the top analogs); see the add-on cells at the end; rank 1 untouched."))
cells.append(md("## Our add-on: generated derivatives of spectral analogs (class-3 candidates)"))
cells.append(code("import os\nos.makedirs('/kaggle/tmp/work', exist_ok=True)\nif not os.path.exists('/content/work'):\n    os.makedirs('/content', exist_ok=True); os.symlink('/kaggle/tmp/work', '/content/work')\nprint('work dir ready')"))
for n in ['fpmodel.py', 'build_train_cache.py', 'fragscore.py', 'analog.py', 'derive.py', 'infer_fp.py', 'gen_addon.py', 'fuse_addon.py']:
    cells.append(code(f'%%writefile /content/work/{n}\n' + open(os.path.join(SRC, n), encoding='utf-8').read()))
cells.append(code(f"""import os, time, json, shutil, subprocess, traceback
import pandas as pd
from pathlib import Path
_root = Path('/kaggle/working'); _ctrl = _root / 'submission_m477_control.csv'
shutil.copy2(_root / 'submission.csv', _ctrl)
_left = 8.8 * 3600 - (time.time() - NB_T0)
GEN_REPORT = dict(status='not run', seconds_left=int(_left))
if _left < 3300:
    GEN_REPORT['status'] = 'skipped: runtime budget'
else:
    try:
        _r = subprocess.run(f'cd /content/work && python -u {ADDON}_addon.py --budget {{int(_left - 900)}} {ADDON_ARGS}', shell=True,
                            capture_output=True, text=True, timeout=max(120, _left - 600))
        print(_r.stdout[-3000:]); print(_r.stderr[-2000:])
        GEN_REPORT = json.load(open(_root / 'gen_addon_report.json')) if _r.returncode == 0 and (_root / 'gen_addon_report.json').is_file() else dict(status='failed', rc=_r.returncode)
    except Exception as exc:
        traceback.print_exc(); GEN_REPORT = dict(status='error', exc=repr(exc))
    if GEN_REPORT.get('inserted') is None:
        shutil.copy2(_ctrl, _root / 'submission.csv'); GEN_REPORT['control_restored'] = True
try:
    _f = pd.read_csv(_root / 'submission.csv', dtype=str, keep_default_na=False)
    assert list(_f.columns) == ['molecule_id', 'smiles'] and _f.molecule_id.tolist() == SAMPLE_IDS and _f.molecule_id.is_unique
    assert _f.smiles.str.split(';').map(len).between(1, 25).all()
    assert _f.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
except Exception:
    traceback.print_exc(); shutil.copy2(_ctrl, _root / 'submission.csv'); GEN_REPORT['control_restored_after_check'] = True
for _p in _root.glob('submission_ours.csv'): _p.unlink()
print('GEN REPORT', GEN_REPORT)
print('FINAL SUBMISSION', _root / 'submission.csv')"""))
nbk = dict(base, cells=cells)
d = os.path.join(S, 'k_' + SLUG.replace('casmi26-', '')); os.makedirs(d, exist_ok=True)
json.dump(nbk, open(os.path.join(d, 'kernel.ipynb'), 'w', encoding='utf-8'), indent=1)
OUR_DS = ['prvsiyan/coconut-casmi26-candidates', 'franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints', 'sergiomalv/casmi-chebi-lmsd-structures',
          'ahmedberatozer/casmi26-pubchem-tier', 'metric/rdkit-2026-3-3-wheel', 'dmitriigluzdov/casmi26-pubchem-popularity-prior']
m = dict(id=f'satyajeetrai/{SLUG}', title=SLUG, code_file='kernel.ipynb', language='python', kernel_type='notebook', is_private=True,
         enable_gpu=True, enable_tpu=False, enable_internet=False, keywords=[],
         dataset_sources=list(dict.fromkeys(meta['dataset_sources'] + OUR_DS)), competition_sources=meta['competition_sources'],
         kernel_sources=['satyajeetrai/casmi26-cpu-prep-cache-pools-baseline', 'satyajeetrai/casmi26-train-fp', 'satyajeetrai/casmi26-pop'],
         model_sources=[], docker_image=meta.get('docker_image'))
json.dump(m, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', d, len(cells), 'cells', m['dataset_sources'])
