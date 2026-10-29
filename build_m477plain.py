import json, glob, os
S = os.path.dirname(os.path.abspath(__file__))
base = json.load(open(glob.glob(os.path.join(S, 'pub/m477/*.ipynb'))[0], encoding='utf8'))
meta = json.load(open(os.path.join(S, 'pub/m477/kernel-metadata.json'), encoding='utf8'))
cells = [dict(c, outputs=[], execution_count=None) if c['cell_type'] == 'code' else c for c in base['cells']]
src = ''.join(cells[1]['source']); assert "COMMIT_MODE = 'smoke'" in src
cells[1]['source'] = src.replace("COMMIT_MODE = 'smoke'", "COMMIT_MODE = 'fast'")
cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": "# Private fork of busyaprime's LB 0.477 notebook (unchanged except COMMIT_MODE = 'fast')\n\nAll credit to the authors listed in section 8; Apache-2.0 code, datasets under their own licences."})
d = os.path.join(S, 'k_m477-plain'); os.makedirs(d, exist_ok=True)
json.dump(dict(base, cells=cells), open(os.path.join(d, 'kernel.ipynb'), 'w', encoding='utf-8'), indent=1)
m = dict(id='satyajeetrai/casmi26-m477-plain', title='casmi26-m477-plain', code_file='kernel.ipynb', language='python', kernel_type='notebook', is_private=True,
         enable_gpu=True, enable_tpu=False, enable_internet=False, keywords=[], dataset_sources=meta['dataset_sources'], competition_sources=meta['competition_sources'],
         kernel_sources=[], model_sources=[], docker_image=meta.get('docker_image'))
json.dump(m, open(os.path.join(d, 'kernel-metadata.json'), 'w'), indent=1); print('wrote', d)
