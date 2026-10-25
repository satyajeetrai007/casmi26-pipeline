"""Train the spectrum -> fingerprint / embedding model on the train cache, holding out the validation structures.
Reports per-epoch retrieval MRR@25 on S1/S2/S3 (class-2 setting: candidates = pool within 10 ppm, scored by
fingerprint log-likelihood + contrastive similarity).
usage: python train_fp.py --out runs/fp_v1 [--epochs 15] [--bs 512] [--layers 6] [--d 512] [--steps_per_epoch 3000]
"""
import os, sys, time, json, argparse, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from fpmodel import *

p = argparse.ArgumentParser()
p.add_argument('--out', default='runs/fp_v1'); p.add_argument('--epochs', type=int, default=15); p.add_argument('--bs', type=int, default=512)
p.add_argument('--layers', type=int, default=6); p.add_argument('--d', type=int, default=512); p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--steps_per_epoch', type=int, default=3000); p.add_argument('--seed', type=int, default=0); p.add_argument('--val_queries', default='/content/work/val_queries_seed0.pkl')
p.add_argument('--w_con', type=float, default=1.0); p.add_argument('--merge_p', type=float, default=0.3); p.add_argument('--no_holdout', action='store_true')
p.add_argument('--resume', default=''); p.add_argument('--eval_every', type=int, default=1); p.add_argument('--max_hours', type=float, default=1e9)
p.add_argument('--init', default='', help='initialise model/molenc weights from an ema.pt (fresh optimiser and schedule)'); p.add_argument('--np_boost', type=float, default=1.0, help='sampling weight multiplier for natural-product libraries')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
dev = 'cuda'; t0 = time.time()
BF16 = torch.cuda.is_bf16_supported(); AMP = torch.bfloat16 if BF16 else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=not BF16)
CACHE = '/content/work/cache'
log = open(f'{args.out}/log.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

# ---------------- data ----------------
meta = pd.read_parquet(f'{CACHE}/spec_meta.parquet')
MZ = np.load(f'{CACHE}/peaks_mz.npy'); IT = np.load(f'{CACHE}/peaks_int.npy'); NPK = np.load(f'{CACHE}/n_peaks.npy')
structs = pd.read_parquet(f'{CACHE}/structs.parquet')
FP_BITS = np.load(f'{CACHE}/fp_morgan4096.npy'); FP_MACCS = np.load(f'{CACHE}/fp_maccs.npy'); FP_CNT = np.load(f'{CACHE}/fp_morgan_count.npy')
EL = np.stack([formula_elements(f) for f in structs.formula_rdkit.values])
sid = meta.struct_id.values.astype(np.int64)
M = encode_meta(meta.adduct.values, meta.ingest_lib.values, meta.instrument_type.values, meta.ionization_mode.values, meta.ce_mean.values, meta.ce_n.values, meta.precursor_mz.values)
meta_key = meta.inchikey14.values; meta_prec = meta.precursor_mz.values
Q = pd.read_pickle(args.val_queries)
holdout_keys = set(Q.key) if not args.no_holdout else set()
train_mask = (~np.isin(meta_key, list(holdout_keys))) & (NPK >= 3) & np.isfinite(meta_prec)
N_META = len(meta); del meta
train_idx = np.where(train_mask)[0]
cnt_per_struct = np.bincount(sid[train_idx], minlength=len(structs))
w = 1.0 / np.sqrt(cnt_per_struct[sid[train_idx]])
if args.np_boost != 1.0:
    NPL = np.isin(M['lib'][train_idx], [LIB_VOCAB.index(x) for x in ('riken', 'gnps', 'massbank', 'mona', 'msdial', 'spectraverse', 'masaryk')])
    w = w * np.where(NPL, args.np_boost, 1.0); P(f'np_boost {args.np_boost}: NP-library share of sampling {w[NPL].sum() / w.sum():.3f}')
w /= w.sum()
P(f'train spectra {len(train_idx)} of {N_META}; held-out structures {len(holdout_keys)}; load {time.time()-t0:.0f}s')
# same-structure/adduct/mode groups for merge augmentation
grp = pd.DataFrame({'s': sid[train_idx], 'a': M['adduct'][train_idx], 'm': M['mode'][train_idx], 'i': train_idx})
grp_key = grp.s * 1000 + grp.a * 2 + grp.m
order = np.argsort(grp_key.values, kind='stable'); gk = grp_key.values[order]; gi = grp.i.values[order]
bounds = np.searchsorted(gk, gk, side='left'); ends = np.searchsorted(gk, gk, side='right')
pos_in_sorted = np.empty(N_META, np.int64); pos_in_sorted[gi] = np.arange(len(gi))


def merge_partner(i, rng):
    ps = pos_in_sorted[i]; lo, hi = bounds[ps], ends[ps]
    if hi - lo < 2:
        return -1
    j = gi[rng.integers(lo, hi)]
    return j if j != i else -1


def make_batch(idx, rng):
    B = len(idx); mz = MZ[idx].astype(np.float32); it = IT[idx].astype(np.float32)
    if args.merge_p > 0:
        do = rng.random(B) < args.merge_p
        partners = np.array([merge_partner(i, rng) if d else -1 for i, d in zip(idx, do)], np.int64)
        sel = np.where(partners >= 0)[0]
        if len(sel):
            j = partners[sel]
            mz2 = np.concatenate([mz[sel], MZ[j].astype(np.float32)], 1); it2 = np.concatenate([it[sel], IT[j].astype(np.float32)], 1)
            top = np.argpartition(-it2, K - 1, axis=1)[:, :K]
            mz[sel] = np.take_along_axis(mz2, top, 1); it[sel] = np.take_along_axis(it2, top, 1)
    valid = it > 0
    drop = (rng.random((B, K)) < 0.1) & valid; drop[np.arange(B), np.argmax(it, 1)] = False; valid &= ~drop
    it = np.where(valid, it * np.exp(rng.normal(0, 0.15, (B, K))).astype(np.float32), 0.0).astype(np.float32)
    it /= np.maximum(it.max(1, keepdims=True), 1e-6)
    mz = np.where(valid, mz * (1 + rng.normal(0, 2e-6, (B, K))).astype(np.float32), np.inf).astype(np.float32)
    order = np.argsort(mz, 1); mz = np.take_along_axis(mz, order, 1); it = np.take_along_axis(it, order, 1)
    npk = valid.sum(1).astype(np.int64); mz[~np.isfinite(mz)] = 0.0
    s = sid[idx]
    bits = np.unpackbits(FP_BITS[s], axis=1)[:, :4096].astype(np.float32); maccs = np.unpackbits(FP_MACCS[s], axis=1)[:, :167].astype(np.float32)
    cnt = FP_CNT[s].astype(np.float32); el = EL[s]
    t = dict(mz=torch.from_numpy(mz), it=torch.from_numpy(it), npk=torch.from_numpy(npk), bits=torch.from_numpy(bits), maccs=torch.from_numpy(maccs),
             cnt=torch.from_numpy(cnt), el=torch.from_numpy(el), s=torch.from_numpy(s))
    for k in ('adduct', 'lib', 'instr', 'mode', 'cef', 'prec'):
        t[k] = torch.from_numpy(M[k][idx])
    return t


# ---------------- validation (class 2 retrieval against the pool) ----------------
pool = pd.read_parquet('/content/work/pools/pool.parquet')
P_BITS = np.load('/content/work/pools/fp_morgan4096.npy'); P_MACCS = np.load('/content/work/pools/fp_maccs.npy'); P_CNT = np.load('/content/work/pools/fp_morgan_count.npy', mmap_mode='r')
pool_mass = pool.mass.values; pm_order = np.argsort(pool_mass); pm_sorted = pool_mass[pm_order]; pool_key = pool.ik14.values
nm_all = meta_prec - M['shift'].astype(np.float64)
VAL = []
for _, q in Q.iterrows():
    idx = q['idx']; target = float(np.median(nm_all[idx]))
    lo, hi = np.searchsorted(pm_sorted, [target * (1 - 10e-6), target * (1 + 10e-6)]); hits = pm_order[lo:hi]
    truth = np.where(pool_key[hits] == q['key'])[0]
    VAL.append(dict(set=q['set'], key=q['key'], idx=np.asarray(idx), cands=hits, truth=int(truth[0]) if len(truth) else -1))
P('val queries', len(VAL), 'truth in pool', np.mean([v['truth'] >= 0 for v in VAL]).round(3), 'median cands', int(np.median([len(v['cands']) for v in VAL])))


def val_batch(idx):
    t = dict(mz=torch.from_numpy(MZ[idx].astype(np.float32)), it=torch.from_numpy(IT[idx].astype(np.float32)), npk=torch.from_numpy(NPK[idx].astype(np.int64)))
    for k in ('adduct', 'lib', 'instr', 'mode', 'cef', 'prec'):
        t[k] = torch.from_numpy(M[k][idx])
    return t


@torch.no_grad()
def evaluate(model, molenc, tag):
    model.eval(); molenc.eval(); rows = []
    for v in VAL:
        if len(v['cands']) == 0:
            rows.append(dict(set=v['set'], rr_fp=0.0, rr_con=0.0, rr_sum=0.0, n=0)); continue
        t = {k: x.to(dev) for k, x in val_batch(v['idx']).items()}
        with torch.autocast('cuda', dtype=AMP):
            o = model(t['mz'], t['it'], t['npk'], t)
        pb = torch.sigmoid(o['bits'].float()); pm = torch.sigmoid(o['maccs'].float()); e = o['emb'].float()
        c = v['cands']
        yb = torch.from_numpy(np.unpackbits(P_BITS[c], axis=1)[:, :4096].astype(np.float32)).to(dev)
        ym = torch.from_numpy(np.unpackbits(P_MACCS[c], axis=1)[:, :167].astype(np.float32)).to(dev)
        yc = torch.from_numpy(P_CNT[c].astype(np.float32)).to(dev)
        ll = (fp_loglik(pb, yb) + fp_loglik(pm, ym)).mean(0)          # [C]
        with torch.autocast('cuda', dtype=AMP):
            me = molenc(yc, ym).float()
        con = (e @ me.T).mean(0)                                        # [C]
        def rr(score):
            order = torch.argsort(-score)[:25].cpu().numpy(); pos = np.where(order == v['truth'])[0]
            return 1.0 / (pos[0] + 1) if len(pos) else 0.0
        z = lambda s: (s - s.mean()) / (s.std() + 1e-6)
        rows.append(dict(set=v['set'], rr_fp=rr(ll), rr_con=rr(con), rr_sum=rr(z(ll) + z(con)), n=len(c)))
    R = pd.DataFrame(rows); g = R.groupby('set')[['rr_fp', 'rr_con', 'rr_sum']].mean().round(4)
    P(f'[{tag}] class-2 MRR@25 by set:\n{g.to_string()}')
    model.train(); molenc.train()
    return g


# ---------------- model ----------------
model = SpecModel(d=args.d, layers=args.layers).to(dev); molenc = MolEncoder(emb=512).to(dev)
NGPU = torch.cuda.device_count(); model_dp = nn.DataParallel(model) if NGPU > 1 else model; molenc_dp = nn.DataParallel(molenc) if NGPU > 1 else molenc
print('GPUs', NGPU, flush=True)
params = list(model.parameters()) + list(molenc.parameters())
P('params M', sum(p.numel() for p in params) / 1e6)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.98))
total = args.epochs * args.steps_per_epoch; warm = min(2000, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total - warm)))))
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}; ema_m = {k: v.detach().clone() for k, v in molenc.state_dict().items()}
start_ep = 0
if args.init and not args.resume:
    ck0 = torch.load(args.init, map_location=dev); model.load_state_dict(ck0['model']); molenc.load_state_dict(ck0['molenc']); P('initialised from', args.init)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}; ema_m = {k: v.detach().clone() for k, v in molenc.state_dict().items()}
if args.resume:
    ck = torch.load(args.resume, map_location=dev); model.load_state_dict(ck['model']); molenc.load_state_dict(ck['molenc']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); ema = ck['ema']; ema_m = ck['ema_m']; start_ep = ck['epoch'] + 1

import threading, queue
def producer(qu, n_steps, seed):
    r = np.random.default_rng(seed)
    for _ in range(n_steps):
        idx = r.choice(train_idx, size=args.bs, replace=True, p=w)
        qu.put(make_batch(np.sort(idx), r))
    qu.put(None)

step = start_ep * args.steps_per_epoch
for ep in range(start_ep, args.epochs):
    qu = queue.Queue(maxsize=6); ths = [threading.Thread(target=producer, args=(qu, args.steps_per_epoch // 3 + 1, args.seed * 1000 + ep * 10 + j), daemon=True) for j in range(3)]
    for th in ths: th.start()
    done = 0; agg = {}; te = time.time(); n_none = 0
    while n_none < 3:
        b = qu.get()
        if b is None:
            n_none += 1; continue
        b = {k: x.to(dev, non_blocking=True) for k, x in b.items()}
        with torch.autocast('cuda', dtype=AMP):
            o = model_dp(b['mz'], b['it'], b['npk'], b); me = molenc_dp(b['cnt'], b['maccs'])
        l_bits = F.binary_cross_entropy_with_logits(o['bits'].float(), b['bits']); l_maccs = F.binary_cross_entropy_with_logits(o['maccs'].float(), b['maccs'])
        l_cnt = F.mse_loss(o['cnt'].float(), torch.log1p(b['cnt'])); l_el = F.mse_loss(o['el'].float(), b['el'] / 10.0)
        logits = (o['emb'].float() @ me.float().T) * model.logit_scale.exp().clamp(max=100)
        same = b['s'][:, None] == b['s'][None, :]; eye = torch.eye(len(b['s']), dtype=torch.bool, device=dev)
        logits = logits.masked_fill(same & ~eye, -1e4)
        tgt = torch.arange(len(b['s']), device=dev); l_con = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
        loss = l_bits * 10 + l_maccs * 3 + l_cnt + l_el + args.w_con * l_con
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); sched.step(); step += 1
        if done % 100 == 0 and (time.time() - t0) / 3600 > args.max_hours:
            P('max_hours reached, stopping epoch early'); break
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(dec).add_(v.detach(), alpha=1 - dec)
                else: ema[k].copy_(v)
            for k, v in molenc.state_dict().items():
                if v.dtype.is_floating_point: ema_m[k].mul_(dec).add_(v.detach(), alpha=1 - dec)
        for k, v in dict(loss=loss, bits=l_bits, maccs=l_maccs, cnt=l_cnt, el=l_el, con=l_con).items():
            agg[k] = agg.get(k, 0.0) + float(v)
        done += 1
        if done % 500 == 0:
            P(f'ep {ep} step {done} ' + ' '.join(f'{k} {v/done:.4f}' for k, v in agg.items()) + f' lr {sched.get_last_lr()[0]:.2e} {time.time()-te:.0f}s')
    P(f'== epoch {ep} done {done} steps ' + ' '.join(f'{k} {v/done:.4f}' for k, v in agg.items()) + f' {time.time()-te:.0f}s total {time.time()-t0:.0f}s')
    torch.save(dict(model=model.state_dict(), molenc=molenc.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ema=ema, ema_m=ema_m, epoch=ep, args=vars(args)), f'{args.out}/last.pt')
    torch.save(dict(model=ema, molenc=ema_m, args=vars(args), epoch=ep), f'{args.out}/ema.pt')
    if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
        cur = {k: v.clone() for k, v in model.state_dict().items()}; cur_m = {k: v.clone() for k, v in molenc.state_dict().items()}
        model.load_state_dict(ema); molenc.load_state_dict(ema_m); evaluate(model, molenc, f'ep{ep} EMA'); model.load_state_dict(cur); molenc.load_state_dict(cur_m)
    if (time.time() - t0) / 3600 > args.max_hours:
        P('max_hours reached after epoch', ep); break
torch.save(dict(model=ema, molenc=ema_m, args=vars(args)), f'{args.out}/ema.pt')
P('TRAIN DONE')
