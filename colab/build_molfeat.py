"""Featurise every pool structure for cross-encoder training (CPU). Output /content/work/molfeat/ (row-aligned with pool.parquet):
mf_atoms.npy [N,64,5] int8, mf_frags.npy [N,64,2] float32, mf_cnt.npy [N,2048] uint8, mf_n.npy [N,2] int16, mf_ok.npy [N] bool.
"""
import os, time, numpy as np, pandas as pd
from multiprocessing import Pool
from molfeat import featurize_many, A_MAX, F_MAX
OUT = '/content/work/molfeat'; os.makedirs(OUT, exist_ok=True); t0 = time.time()
pool = pd.read_parquet('/content/work/pools/pool.parquet', columns=['ik14', 'smiles']); N = len(pool); smi = pool.smiles.tolist()
A = np.lib.format.open_memmap(f'{OUT}/mf_atoms.npy', 'w+', np.int8, (N, A_MAX, 5)); Fr = np.lib.format.open_memmap(f'{OUT}/mf_frags.npy', 'w+', np.float32, (N, F_MAX, 2))
C = np.lib.format.open_memmap(f'{OUT}/mf_cnt.npy', 'w+', np.uint8, (N, 2048)); NN = np.zeros((N, 2), np.int16); OK = np.zeros(N, bool)
step = 2000; chunks = [(i, smi[i:i + step]) for i in range(0, N, step)]
def job(a):
    i, s = a; return (i,) + featurize_many(s)
with Pool(int(os.environ.get('NPROC', '4'))) as p:
    for j, (i, a, f, c, nn, ok) in enumerate(p.imap_unordered(job, chunks)):
        A[i:i + len(ok)] = a; Fr[i:i + len(ok)] = f; C[i:i + len(ok)] = c; NN[i:i + len(ok)] = nn; OK[i:i + len(ok)] = ok
        if j % 50 == 0: print(f'  {j}/{len(chunks)} {time.time()-t0:.0f}s', flush=True)
A.flush(); Fr.flush(); C.flush(); np.save(f'{OUT}/mf_n.npy', NN); np.save(f'{OUT}/mf_ok.npy', OK)
print('pool', N, 'ok', int(OK.sum()), 'median atoms', int(np.median(NN[:, 0])), 'median frags', int(np.median(NN[:, 1])), f'{time.time()-t0:.0f}s')
print('MOLFEAT DONE')
