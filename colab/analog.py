"""Mass-shifted analog search (CPU, numba).

For a query molecule, compare its merged spectrum (per polarity) with one representative spectrum of every training
structure within +-win Da. A query peak matches a reference peak either directly (same m/z) or shifted by the
precursor mass difference (the fragment that carries the modification), as in modified cosine. The best-matching
relatives (analogs) then score candidates by structural similarity: ap = max_j Tanimoto(candidate, analog_j) * sim_j^4.
Natural products come in families (methyl / hydroxyl / glycoside variants), so a relative's spectrum is strong
evidence for the candidate that looks like it, even when the true structure has no public spectrum.
"""
import numpy as np, pandas as pd
from numba import njit, prange

MAXP = 64


@njit(cache=True)
def _top_peaks(mz, it, n, maxp, out_mz, out_w):
    """Top-maxp peaks by intensity of one spectrum, m/z-sorted, sqrt intensities scaled to unit norm. Returns count."""
    k = min(n, maxp)
    if k == 0:
        return 0
    order = np.argsort(-it[:n])[:k]
    sel_mz = mz[order]; sel_w = np.sqrt(np.maximum(it[order], 0.0))
    o2 = np.argsort(sel_mz)
    nrm = 0.0
    for j in range(k):
        nrm += sel_w[j] * sel_w[j]
    nrm = np.sqrt(nrm) + 1e-12
    for j in range(k):
        out_mz[j] = sel_mz[o2[j]]; out_w[j] = sel_w[o2[j]] / nrm
    return k


@njit(parallel=True, cache=True)
def pack_reps(MZ, IT, NPK, rows, maxp, R_mz, R_w, R_n):
    for k in prange(len(rows)):
        r = rows[k]
        R_n[k] = _top_peaks(MZ[r].astype(np.float64), IT[r].astype(np.float64), NPK[r], maxp, R_mz[k], R_w[k])


@njit(cache=True)
def _sim_one(qm, qw, qord, nq, rm, rw, nr, shift, tol):
    """Greedy shifted cosine: query peaks in descending weight; each reference peak used once."""
    used = np.zeros(nr, np.bool_)
    s = 0.0
    use_shift = abs(shift) > 2 * tol
    for t in range(nq):
        i = qord[t]; x = qm[i]
        best = -1; bw = 0.0
        lo = np.searchsorted(rm[:nr], x - tol)
        j = lo
        while j < nr and rm[j] <= x + tol:
            if not used[j] and rw[j] > bw:
                bw = rw[j]; best = j
            j += 1
        if use_shift:
            y = x - shift
            lo = np.searchsorted(rm[:nr], y - tol)
            j = lo
            while j < nr and rm[j] <= y + tol:
                if not used[j] and rw[j] > bw:
                    bw = rw[j]; best = j
                j += 1
        if best >= 0:
            used[best] = True; s += qw[i] * bw
    return s


@njit(parallel=True, cache=True)
def sim_many(qm, qw, qord, nq, R_mz, R_w, R_n, ridx, shifts, tol, out):
    for k in prange(len(ridx)):
        r = ridx[k]
        out[k] = _sim_one(qm, qw, qord, nq, R_mz[r], R_w[r], R_n[r], shifts[k], tol)


def merge_peaks(specs, tol=0.005):
    """Union of the spectra (each scaled to base peak 1), near-duplicate m/z merged by max intensity."""
    mz = np.concatenate([np.asarray(s['mz'], np.float64) for s in specs]) if specs else np.zeros(0)
    it = np.concatenate([np.asarray(s['it'], np.float64) / max(float(np.max(s['it'])), 1e-12) if len(s['it']) else np.zeros(0) for s in specs]) if specs else np.zeros(0)
    if len(mz) == 0:
        return mz, it
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool); last = 0
    for j in range(1, len(mz)):
        if mz[j] - mz[last] < tol:
            if it[j] >= it[last]:
                keep[last] = False; last = j
            else:
                keep[j] = False
        else:
            last = j
    return mz[keep], it[keep]


class AnalogIndex:
    """One representative spectrum (most peaks) per (structure, polarity, library), sorted by neutral mass."""

    def __init__(self, MZ, IT, NPK, keys, modes, libs, nm, ok, maxp=MAXP):
        idx = np.where(ok)[0]
        df = pd.DataFrame(dict(i=idx, k=keys[idx], m=modes[idx], l=libs[idx], n=np.asarray(NPK)[idx]))
        df = df.sort_values('n', ascending=False, kind='stable').drop_duplicates(['k', 'm', 'l'])
        rows = df.i.values[np.argsort(nm[df.i.values], kind='stable')]
        self.rows = rows; self.nm = nm[rows].astype(np.float64); self.key = keys[rows]; self.lib = libs[rows]
        self.pos = (modes[rows] == 'positive')
        n = len(rows)
        self.R_mz = np.zeros((n, maxp), np.float64); self.R_w = np.zeros((n, maxp), np.float64); self.R_n = np.zeros(n, np.int64)
        for c0 in range(0, n, 200000):                      # chunked gather from the memory-mapped cache
            r = rows[c0:c0 + 200000]; o = np.argsort(r); rs = r[o]
            mz = np.asarray(MZ[rs], np.float64); it = np.asarray(IT[rs], np.float64); npk = np.asarray(NPK)[rs].astype(np.int64)
            tm = np.zeros((len(rs), maxp)); tw = np.zeros((len(rs), maxp)); tn = np.zeros(len(rs), np.int64)
            pack_reps(mz, it, npk, np.arange(len(rs)), maxp, tm, tw, tn)
            inv = np.empty(len(o), np.int64); inv[o] = np.arange(len(o))
            self.R_mz[c0:c0 + len(r)] = tm[inv]; self.R_w[c0:c0 + len(r)] = tw[inv]; self.R_n[c0:c0 + len(r)] = tn[inv]
        self.maxp = maxp

    def search(self, specs, target, win=200.0, tol=0.01, n_out=100, exclude_key=None, exclude_lib=None):
        """Returns list of (key, sim, shift) of the n_out best analogs (best representative per structure)."""
        best = {}
        lo, hi = np.searchsorted(self.nm, [target - win, target + win])
        for pos in (True, False):
            grp = [s for s in specs if (s['mode'] == 'positive') == pos]
            if not grp:
                continue
            mz, it = merge_peaks(grp)
            qm = np.zeros(self.maxp); qw = np.zeros(self.maxp)
            nq = _top_peaks(mz, it, len(mz), self.maxp, qm, qw)
            if nq == 0:
                continue
            qord = np.argsort(-qw[:nq]).astype(np.int64)
            ridx = np.arange(lo, hi)[self.pos[lo:hi] == pos]
            if exclude_key is not None and len(ridx):
                bad = self.key[ridx] == exclude_key
                if exclude_lib is not None:
                    bad &= self.lib[ridx] == exclude_lib
                ridx = ridx[~bad]
            if len(ridx) == 0:
                continue
            shifts = target - self.nm[ridx]; out = np.zeros(len(ridx))
            sim_many(qm, qw, qord, nq, self.R_mz, self.R_w, self.R_n, ridx, shifts, tol, out)
            top = np.argsort(-out)[:4 * n_out]
            for j in top:
                k = self.key[ridx[j]]; v = float(out[j])
                if v > best.get(k, (-1.0, 0.0))[0]:
                    best[k] = (v, float(shifts[j]))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:n_out]
        return [(k, v[0], v[1]) for k, v in items]


def analog_features(cand_bits, an_bits, sims, power=4.0):
    """cand_bits (n, B) float32 0/1, an_bits (m, B), sims (m,). Returns dict of per-candidate features."""
    n = len(cand_bits)
    if len(an_bits) == 0 or n == 0:
        z = np.zeros(n, np.float32)
        return dict(ap=z, ap1=z, ap_best_tan=z, ap_top_tan=z, ap_mean_tan=z)
    inter = cand_bits @ an_bits.T
    tan = inter / (cand_bits.sum(1)[:, None] + an_bits.sum(1)[None, :] - inter + 1e-9)
    w = np.asarray(sims, np.float32) ** power
    return dict(ap=(tan * w[None]).max(1).astype(np.float32), ap1=(tan * sims[None]).max(1).astype(np.float32),
                ap_best_tan=tan.max(1).astype(np.float32), ap_top_tan=tan[:, 0].astype(np.float32),
                ap_mean_tan=((tan * w[None]).sum(1) / (w.sum() + 1e-9)).astype(np.float32))
