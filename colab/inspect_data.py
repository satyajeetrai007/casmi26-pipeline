import os, sys, glob, pickle, json, numpy as np, pandas as pd, pyarrow.parquet as pq
D = '/content/data'
os.makedirs('/content/work', exist_ok=True)
for f in ['comp/train.parquet', 'comp/test.parquet']:
    p = f'{D}/{f}'; pf = pq.ParquetFile(p)
    print('==', f, 'rows', pf.metadata.num_rows, 'row_groups', pf.num_row_groups); print(pf.schema_arrow)
te = pd.read_parquet(f'{D}/comp/test.parquet'); print(te.head(3).T); print('test molecules', te.molecule_id.nunique())
cols = ['ingest_lib', 'inchikey14', 'adduct', 'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_ev',
        'collision_energy_orig_units', 'molecular_formula', 'num_peaks', 'precursor_error_ppm', 'base_peak_intensity', 'normalized_smiles']
meta = pd.read_parquet(f'{D}/comp/train.parquet', columns=cols)
print(meta.groupby('ingest_lib').agg(n=('inchikey14', 'size'), structs=('inchikey14', 'nunique')))
print('adducts top20'); print(meta.adduct.value_counts().head(20))
print('units'); print(meta.collision_energy_orig_units.value_counts(dropna=False))
print('ppm err quantiles by lib'); print(meta.groupby('ingest_lib').precursor_error_ppm.describe()[['25%', '50%', '75%']])
print('num_peaks q'); print(meta.num_peaks.describe())
pub = set(meta.loc[~meta.ingest_lib.isin(['enveda-180', 'enveda-np-examples']), 'inchikey14'])
e180 = set(meta.loc[meta.ingest_lib == 'enveda-180', 'inchikey14']); npx = set(meta.loc[meta.ingest_lib == 'enveda-np-examples', 'inchikey14'])
print('public structs', len(pub), 'enveda-180 structs', len(e180), 'e180 with public spectra', len(e180 & pub), 'np-examples', len(npx), 'npx in public', len(npx & pub))
meta.to_parquet('/content/work/train_meta.parquet')
for root in sorted(glob.glob(f'{D}/ds/*')):
    print('#####', root)
    for p in sorted(glob.glob(root + '/**/*', recursive=True)):
        if os.path.isdir(p):
            continue
        sz = os.path.getsize(p); name = os.path.relpath(p, root)
        try:
            if p.endswith('.npy'):
                a = np.load(p, mmap_mode='r'); print(f'  {name} {sz/1e6:.1f}MB npy {a.shape} {a.dtype}')
            elif p.endswith('.pkl'):
                o = pickle.load(open(p, 'rb'))
                if isinstance(o, dict):
                    print(f'  {name} {sz/1e6:.1f}MB dict keys {list(o.keys())[:20]}',
                          {k: (type(v).__name__, getattr(v, 'shape', None) or (len(v) if hasattr(v, '__len__') else None)) for k, v in list(o.items())[:12]})
                elif isinstance(o, pd.DataFrame):
                    print(f'  {name} {sz/1e6:.1f}MB DataFrame {o.shape} cols {list(o.columns)}'); print(o.head(2).T)
                else:
                    print(f'  {name} {sz/1e6:.1f}MB {type(o).__name__} len', len(o) if hasattr(o, '__len__') else '?', str(o)[:300])
            elif p.endswith('.parquet'):
                pf = pq.ParquetFile(p); print(f'  {name} {sz/1e6:.1f}MB parquet rows {pf.metadata.num_rows}'); print(pf.schema_arrow); print(pd.read_parquet(p).head(2).T)
            elif sz < 20000 and p.split('.')[-1] in ('json', 'md', 'txt', 'csv'):
                print(f'  {name} {sz}B:\n' + open(p, errors='replace').read()[:3000])
            else:
                print(f'  {name} {sz/1e6:.1f}MB')
        except Exception as e:
            print('  ERR', name, repr(e))
print('INSPECT DONE')
