"""Train the spectrum x candidate cross-encoder on train spectra with hard same-formula negatives from the pool.
Validation structures (S1/S2/S3 of val_queries_seed0.pkl) are excluded from training as truths AND as negatives.
Each epoch reports MRR@25 on validation molecules against candidate lists = truth + same-formula pool structures
(+ mass neighbours, up to 50), for the cross-encoder and for the spectrum model's fingerprint log-likelihood.
usage: python train_xenc.py --init runs/fp_v2/ema.pt --out runs/xenc_v1 [--epochs 8 --steps 5000 --bs 32 --group 16 --max_hours 5]
"""
import os, sys, time, math, argparse, threading, queue, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from fpmodel import *
from xenc import XEnc

p = argparse.ArgumentParser()
p.add_argument('--init', required=True); p.add_argument('--out', default='runs/xenc_v1'); p.add_argument('--resume', default='')
p.add_argument('--epochs', type=int, default=8); p.add_argument('--steps', type=int, default=5000); p.add_argument('--bs', type=int, default=32); p.add_argument('--group', type=int, default=16)
p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--lr_spec', type=float, default=5e-5); p.add_argument('--max_hours', type=float, default=5.0)
p.add_argument('--n_formula', type=int, default=11, help='negatives from the same formula (rest: mass neighbours / random)'); p.add_argument('--seed', type=int, default=0)
p.add_argument('--layers', type=int, default=3); p.add_argument('--d', type=int, default=256); p.add_argument('--val_max', type=int, default=900)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True); torch.manual_seed(args.seed); dev = 'cuda'; t0 = time.time()
BF16 = torch.cuda.is_bf16_supported(); AMP = torch.bfloat16 if BF16 else torch.float16; scaler = torch.amp.GradScaler('cuda', enabled=not BF16)
log = open(f'{args.out}/log.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

CACHE = '/content/work/cache'; POOLS = '/content/work/pools'; MF = '/content/work/molfeat'
meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet', columns=['inchikey14', 'adduct', 'ingest_lib', 'instrument_type', 'ionization_mode', 'ce_mean', 'ce_n', 'precursor_mz', 'struct_id'])
MZ = np.load(f'{CACHE}/peaks_mz.npy'); IT = np.load(f'{CACHE}/peaks_int.npy'); NPK = np.load(f'{CACHE}/n_peaks.npy')
M = encode_meta(meta.adduct.values, meta.ingest_lib.values, meta.instrument_type.values, meta.ionization_mode.values, meta.ce_mean.values, meta.ce_n.values, meta.precursor_mz.values)
pool = pd.read_parquet(f'{POOLS}/pool.parquet', columns=['ik14', 'mass', 'formula', 'in_train'])
MF_A = np.load(f'{MF}/mf_atoms.npy', mmap_mode='r'); MF_F = np.load(f'{MF}/mf_frags.npy', mmap_mode='r'); MF_C = np.load(f'{MF}/mf_cnt.npy', mmap_mode='r'); MF_N = np.load(f'{MF}/mf_n.npy'); MF_OK = np.load(f'{MF}/mf_ok.npy')
P_BITS = np.load(f'{POOLS}/fp_morgan4096.npy', mmap_mode='r'); P_MACCS = np.load(f'{POOLS}/fp_maccs.npy', mmap_mode='r')
row_of = pd.Series(np.arange(len(pool)), index=pool.ik14.values); row_of = row_of[~row_of.index.duplicated()]
Q = pd.read_pickle('/content/work/val_queries_seed0.pkl'); hold = set(Q.key)
spec_row = row_of.reindex(meta.inchikey14.values).values                      # pool row of each spectrum's structure (NaN if absent)
ok_spec = np.isfinite(spec_row) & (NPK >= 3) & np.isfinite(meta.precursor_mz.values) & ~meta.inchikey14.isin(hold).values
spec_row = np.nan_to_num(spec_row, nan=-1).astype(np.int64); ok_spec &= spec_row >= 0
ok_spec[ok_spec] &= MF_OK[spec_row[ok_spec]]
train_idx = np.where(ok_spec)[0]
sid = meta.struct_id.values; cnt_s = np.bincount(sid[train_idx], minlength=sid.max() + 1); w = 1.0 / np.sqrt(cnt_s[sid[train_idx]]); w /= w.sum()
# negatives: same formula groups and mass order, excluding validation structures and unfeaturisable rows
# negatives ONLY from other training structures: positives are always training structures, so external (COCONUT/PubChem)
# negatives would let the model learn 'external SMILES style = wrong', which inverts at test time (class-2 answers are external)
neg_ok = MF_OK & ~pool.ik14.isin(hold).values & pool.in_train.values.astype(bool)
fcode = pd.factorize(pool.formula.values)[0]; order_f = np.lexsort((np.arange(len(pool)), fcode)); fs = fcode[order_f]
g_lo = np.searchsorted(fs, fcode, 'left'); g_hi = np.searchsorted(fs, fcode, 'right')
order_m = np.argsort(pool.mass.values); pos_m = np.empty(len(pool), np.int64); pos_m[order_m] = np.arange(len(pool))
neg_rows_all = np.where(neg_ok)[0]
P(f'train spectra {len(train_idx)} | pool {len(pool)} featurised {int(MF_OK.sum())} | load {time.time()-t0:.0f}s')


def negatives(r, k, rng):
    out = []
    lo, hi = g_lo[r], g_hi[r]
    if hi - lo > 1:
        cand = order_f[lo:hi]; cand = cand[(cand != r) & neg_ok[cand]]
        if len(cand): out = list(rng.choice(cand, size=min(args.n_formula, len(cand)), replace=False))
    pm = pos_m[r]; tries = 0
    while len(out) < k - 2 and tries < 60:                                      # mass neighbours (+-50 rows ~ a few ppm)
        j = order_m[np.clip(pm + rng.integers(-60, 61), 0, len(pool) - 1)]; tries += 1
        if j != r and neg_ok[j] and j not in out: out.append(j)
    while len(out) < k:
        j = neg_rows_all[rng.integers(len(neg_rows_all))]
        if j != r: out.append(j)
    return out[:k]


def spec_tensors(idx, rng=None, aug=False):
    B = len(idx); mz = MZ[idx].astype(np.float32); it = IT[idx].astype(np.float32)
    if aug:
        valid = it > 0; drop = (rng.random((B, K)) < 0.1) & valid; drop[np.arange(B), np.argmax(it, 1)] = False; valid &= ~drop
        it = np.where(valid, it * np.exp(rng.normal(0, 0.15, (B, K))).astype(np.float32), 0.0).astype(np.float32); it /= np.maximum(it.max(1, keepdims=True), 1e-6)
        mz = np.where(valid, mz, np.inf).astype(np.float32); o = np.argsort(mz, 1); mz = np.take_along_axis(mz, o, 1); it = np.take_along_axis(it, o, 1); mz[~np.isfinite(mz)] = 0
        npk = valid.sum(1).astype(np.int64)
    else:
        npk = NPK[idx].astype(np.int64)
    t = dict(mz=torch.from_numpy(mz), it=torch.from_numpy(it), npk=torch.from_numpy(npk))
    for k_ in ('adduct', 'lib', 'instr', 'mode', 'cef', 'prec'):
        t[k_] = torch.from_numpy(M[k_][idx])
    return t


def cand_tensors(rows):                                                       # rows [B, G]
    flat = rows.reshape(-1); srt = np.argsort(flat); inv = np.argsort(srt); fs_ = flat[srt]
    A = MF_A[fs_][inv]; Fr = MF_F[fs_][inv]; C = MF_C[fs_][inv]; N = MF_N[flat]
    B, G = rows.shape
    return dict(atoms=torch.from_numpy(A.reshape(B, G, *A.shape[1:])), frags=torch.from_numpy(Fr.reshape(B, G, *Fr.shape[1:])),
                nn_=torch.from_numpy(N.reshape(B, G, 2)), cnt=torch.from_numpy(C.reshape(B, G, 2048)))


def make_batch(rng):
    idx = np.sort(rng.choice(train_idx, size=args.bs, replace=True, p=w))
    rows = np.zeros((args.bs, args.group), np.int64)
    for b, i in enumerate(idx):
        r = spec_row[i]; rows[b, 0] = r; rows[b, 1:] = negatives(r, args.group - 1, rng)
    perm = np.argsort(rng.random((args.bs, args.group)), 1); rows = np.take_along_axis(rows, perm, 1); label = np.argmax(perm == 0, 1)
    t = spec_tensors(idx, rng, aug=True); t.update(cand_tensors(rows)); t['label'] = torch.from_numpy(label)
    return t


# ---------------- validation lists
nm_all = meta.precursor_mz.values - M['shift'].astype(np.float64)
VAL = []
vrng = np.random.default_rng(123)
for _, q in Q.iterrows():
    if q['key'] not in row_of.index: continue
    r = int(row_of[q['key']])
    if not MF_OK[r]: continue
    lo, hi = g_lo[r], g_hi[r]; same = order_f[lo:hi]; same = same[(same != r) & MF_OK[same]]
    pm = pos_m[r]; near = order_m[max(0, pm - 40):pm + 40]; near = near[(near != r) & MF_OK[near] & (fcode[near] != fcode[r])]
    c = np.concatenate([[r], same[:39], near[:max(0, 49 - min(len(same), 39))]]).astype(np.int64)
    VAL.append(dict(set=q['set'], idx=np.asarray(q['idx']), rows=c, n_same=len(same)))
vsel = vrng.permutation(len(VAL))[:args.val_max]; VAL = [VAL[i] for i in sorted(vsel)]
P('val molecules', len(VAL), 'median list', int(np.median([len(v['rows']) for v in VAL])), 'median same-formula', int(np.median([v['n_same'] for v in VAL])))

# ---------------- model
ck = torch.load(args.init, map_location='cpu'); a = ck.get('args', {})
spec = SpecModel(d=a.get('d', 512), layers=a.get('layers', 6)); spec.load_state_dict(ck['model'])
model = XEnc(spec, spec_d=a.get('d', 512), d=args.d, layers=args.layers).to(dev)
NG = torch.cuda.device_count(); model_dp = nn.DataParallel(model) if NG > 1 else model
spec_params = list(model.spec.parameters()); other = [p_ for n_, p_ in model.named_parameters() if not n_.startswith('spec.')]
opt = torch.optim.AdamW([{'params': spec_params, 'lr': args.lr_spec}, {'params': other, 'lr': args.lr}], weight_decay=0.05, betas=(0.9, 0.98))
total = args.epochs * args.steps; warm = min(1500, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}; start_ep = 0
if args.resume:
    ckr = torch.load(args.resume, map_location=dev); model.load_state_dict(ckr['model']); opt.load_state_dict(ckr['opt']); sched.load_state_dict(ckr['sched']); ema = ckr['ema']; start_ep = ckr['epoch'] + 1
P('GPUs', NG, 'params M', round(sum(p_.numel() for p_ in model.parameters()) / 1e6, 1), 'cross part M', round(sum(p_.numel() for p_ in other) / 1e6, 1))


@torch.no_grad()
def evaluate(tag, baseline=False):
    model.eval(); rows_out = []
    for v in VAL:
        t = spec_tensors(v['idx']); t = {k_: x.to(dev) for k_, x in t.items()}
        G = len(v['rows']); S = len(v['idx'])
        rows = np.broadcast_to(v['rows'], (S, G)).copy(); ct = {k_: x.to(dev) for k_, x in cand_tensors(rows).items()}
        with torch.autocast('cuda', dtype=AMP):
            sc = model(t['mz'], t['it'], t['npk'], t['adduct'], t['lib'], t['instr'], t['mode'], t['cef'], t['prec'], ct['atoms'], ct['frags'], ct['nn_'], ct['cnt']).float().mean(0)
        rr = lambda s: next((1.0 / (i + 1) for i, j in enumerate(torch.argsort(-s)[:25].tolist()) if j == 0), 0.0)
        row = dict(set=v['set'], xenc=rr(sc))
        if baseline:
            with torch.autocast('cuda', dtype=AMP):
                o = model.spec(t['mz'], t['it'], t['npk'], t)
            pb = torch.sigmoid(o['bits'].float()); pm = torch.sigmoid(o['maccs'].float())
            yb = torch.from_numpy(np.unpackbits(P_BITS[v['rows']], axis=1)[:, :4096].astype(np.float32)).to(dev); ym = torch.from_numpy(np.unpackbits(P_MACCS[v['rows']], axis=1)[:, :167].astype(np.float32)).to(dev)
            row['fp_ll'] = rr((fp_loglik(pb, yb) + fp_loglik(pm, ym)).mean(0))
        rows_out.append(row)
    D = pd.DataFrame(rows_out); P(f'[{tag}] MRR@25 on truth + same-formula lists:\n' + D.groupby('set').mean(numeric_only=True).round(4).to_string()); model.train()
    return D


evaluate('init', baseline=True)


def producer(qu, n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n): qu.put(make_batch(rng))
    qu.put(None)


step = start_ep * args.steps
for ep in range(start_ep, args.epochs):
    qu = queue.Queue(maxsize=8); ths = [threading.Thread(target=producer, args=(qu, args.steps // 3 + 1, args.seed * 997 + ep * 7 + j), daemon=True) for j in range(3)]
    for th in ths: th.start()
    done = 0; lsum = 0.0; acc = 0.0; te = time.time(); nn_none = 0
    while nn_none < 3:
        b = qu.get()
        if b is None: nn_none += 1; continue
        b = {k_: x.to(dev, non_blocking=True) for k_, x in b.items()}
        with torch.autocast('cuda', dtype=AMP):
            sc = model_dp(b['mz'], b['it'], b['npk'], b['adduct'], b['lib'], b['instr'], b['mode'], b['cef'], b['prec'], b['atoms'], b['frags'], b['nn_'], b['cnt']).float()
        loss = F.cross_entropy(sc, b['label'])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for k_, v_ in model.state_dict().items():
                if v_.dtype.is_floating_point: ema[k_].mul_(dec).add_(v_.detach(), alpha=1 - dec)
                else: ema[k_].copy_(v_)
        lsum += float(loss); acc += float((sc.argmax(1) == b['label']).float().mean()); done += 1
        if done % 500 == 0:
            P(f'ep {ep} step {done} loss {lsum/done:.4f} top1 {acc/done:.3f} lr {sched.get_last_lr()[1]:.2e} {time.time()-te:.0f}s')
        if done % 100 == 0 and (time.time() - t0) / 3600 > args.max_hours:
            P('max_hours reached'); break
    P(f'== epoch {ep} done {done} steps loss {lsum/max(done,1):.4f} top1 {acc/max(done,1):.3f} {time.time()-te:.0f}s total {time.time()-t0:.0f}s')
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ema=ema, epoch=ep, args=vars(args), spec_args=a), f'{args.out}/last.pt')
    torch.save(dict(model=ema, args=vars(args), spec_args=a, epoch=ep), f'{args.out}/ema.pt')
    cur = {k_: v_.clone() for k_, v_ in model.state_dict().items()}; model.load_state_dict(ema); evaluate(f'ep{ep} EMA'); model.load_state_dict(cur)
    if (time.time() - t0) / 3600 > args.max_hours:
        P('max_hours reached after epoch', ep); break
P('TRAIN XENC DONE')
