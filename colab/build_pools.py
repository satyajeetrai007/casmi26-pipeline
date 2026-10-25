"""Merge candidate structure pools (train structures + COCONUT + PubChemLite + ChEBI/LMSD) into one table
keyed by InChIKey14, with RDKit-computed neutral mono mass, formula and fingerprints.
Output /content/work/pools/: pool.parquet (ik14, smiles, mass, formula, src_mask, in_train), fp_morgan4096.npy (packed),
fp_maccs.npy (packed), fp_morgan_count.npy (2048 uint8).
src_mask bits: 1 train, 2 coconut, 4 pubchemlite, 8 chebi_lmsd.
"""
import os, pickle, time, numpy as np, pandas as pd
from multiprocessing import Pool
from build_train_cache import fp_worker

D = '/content/data/ds'; OUT = '/content/work/pools'; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
structs = pd.read_parquet('/content/work/cache/structs.parquet')
rows = {}
def add(keys, smiles, bit):
    n_new = 0
    for k, s in zip(keys, smiles):
        k = str(k)[:14]
        if k in rows:
            rows[k][1] |= bit
        else:
            rows[k] = [str(s), bit]; n_new += 1
    return n_new
print('train', add(structs.inchikey14.values, structs.normalized_smiles.values, 1))
m = pickle.load(open(f'{D}/coconut-casmi26-candidates/coco_meta.pkl', 'rb')); print('coconut new', add(m['keys'], m['smiles'], 2))
m = pickle.load(open(f'{D}/casmi26-pubchemlite-cc0-candidate-fingerprints/pubchemlite_meta.pkl', 'rb')); print('pubchemlite new', add(m['keys'], m['smiles'], 4))
m = pickle.load(open(f'{D}/casmi-chebi-lmsd-structures/bio_meta.pkl', 'rb')); print('chebi/lmsd new', add(m['keys'], m['smiles'], 8))
keys = list(rows.keys()); smi = [rows[k][0] for k in keys]; src = np.array([rows[k][1] for k in keys], np.int8)
print('total pool', len(keys), f'{time.time()-t0:.0f}s', flush=True)
chunks = [smi[i:i + 2000] for i in range(0, len(smi), 2000)]
bits, maccs, cnt, mass, formula, ok, ik14 = [], [], [], [], [], [], []
with Pool(int(os.environ.get('NPROC', '12'))) as pool:
    for j, r in enumerate(pool.imap(fp_worker, chunks)):
        bits.append(r[0]); maccs.append(r[1]); cnt.append(r[2]); mass.append(r[3]); formula += r[4]; ok.append(r[5]); ik14 += r[6]
        if j % 50 == 0:
            print(f'  chunk {j}/{len(chunks)} {time.time()-t0:.0f}s', flush=True)
bits = np.concatenate(bits); maccs = np.concatenate(maccs); cnt = np.concatenate(cnt)
df = pd.DataFrame({'orig_key': keys, 'smiles': smi, 'mass': np.concatenate(mass), 'formula': formula, 'src_mask': src, 'rdkit_ok': np.concatenate(ok), 'ik14': ik14})
df['ik14'] = np.where(df.ik14.str.len() == 14, df.ik14, df.orig_key)       # RDKit failures keep the source key
# re-key by the computed InChIKey14: the public COCONUT file's keys are not InChIKey14s (0 overlap with train)
df['row'] = np.arange(len(df)); df['is_train'] = (df.src_mask & 1) > 0
df = df.sort_values(['ik14', 'is_train'], ascending=[True, False])
mask_or = df.groupby('ik14').src_mask.agg(lambda v: int(np.bitwise_or.reduce(v.values)))
first = df.drop_duplicates('ik14', keep='first').copy(); first['src_mask'] = first.ik14.map(mask_or).astype(np.int8)
print('pool rows', len(df), '-> unique InChIKey14', len(first), '| train keys matched by rdkit', int((first.is_train & (first.orig_key == first.ik14)).sum()))
print('src_mask counts', first.src_mask.value_counts().to_dict())
sel = first.row.values; np.save(f'{OUT}/fp_morgan4096.npy', bits[sel]); np.save(f'{OUT}/fp_maccs.npy', maccs[sel]); np.save(f'{OUT}/fp_morgan_count.npy', cnt[sel])
first['in_train'] = first.is_train
first = first.drop(columns=['row', 'is_train']).reset_index(drop=True); first.to_parquet(f'{OUT}/pool.parquet'); df = first
print('rdkit failures', int((~df.rdkit_ok).sum()), 'mass nan', int(df.mass.isna().sum()), f'{time.time()-t0:.0f}s')
print('POOLS DONE')
