"""Spectrum transformer -> fingerprint / embedding model (shared by training and inference)."""
import math, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

K = 128
ELEMENTS = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
ADDUCT_VOCAB = ['[M+H]+', '[M-H]-', '[2M+Na]+', '[M+Na]+', '[2M+H]+', '[M+CH2O2-H]-', '[2M-H]-', '[M+NH4]+', '[M+Cl]-', '[M-H2O+H]+',
                '[M+K]+', '[M]+', '[M+C2H4O2-H]-', '[M-2H2O+H]+', '[M+2H]2+', '[2M+Na-2H]-', '[2M+CH2O2-H]-', '[M-H2O]+', '[M-CH3]-', '[M-2H]-']
LIB_VOCAB = ['enveda-180', 'enveda-np-examples', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse', 'msdial', 'drug_plus', 'masaryk']
INSTR_CLASSES = ['tims', 'orbitrap', 'qtof', 'other', 'unknown']
PROTON = 1.007276
ADDUCT_SHIFT = {'[M+H]+': PROTON, '[M+NH4]+': 18.033823, '[M-H2O+H]+': PROTON - 18.010565, '[M-2H2O+H]+': PROTON - 2 * 18.010565,
                '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -PROTON, '[M-H2O-H]-': -PROTON - 18.010565,
                '[M+CH2O2-H]-': 46.005479 - PROTON, '[M+Cl]-': 34.969402, '[2M+H]+': PROTON, '[2M+Na]+': 22.989218, '[2M-H]-': -PROTON,
                '[M]+': 0.0, '[M+C2H4O2-H]-': 60.021129 - PROTON, '[M+2H]2+': PROTON, '[2M+Na-2H]-': 22.989218 - 2 * PROTON,
                '[2M+CH2O2-H]-': 46.005479 - PROTON, '[M-H2O]+': -18.010565, '[M-CH3]-': -15.023475, '[M-2H]-': -2 * PROTON}


def instr_class(s):
    s = (s or '').lower()
    if s == '' or s == 'none':
        return 4
    if 'tims' in s:
        return 0
    if any(w in s for w in ('orbitrap', 'exactive', 'lumos', 'velos', 'fusion', 'hcd', 'ltq', 'astral', 'exploris')):
        return 1
    if 'tof' in s:
        return 2
    return 3


def formula_elements(f):
    out = np.zeros(len(ELEMENTS), np.float32)
    if not isinstance(f, str):
        return out
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        if el in ELEMENTS:
            out[ELEMENTS.index(el)] += int(n) if n else 1
    return out


def encode_meta(adduct, lib, instr, mode, ce_mean, ce_n, prec):
    """-> dict of int/float numpy arrays for a batch (lists/arrays of length B)."""
    a = np.array([ADDUCT_VOCAB.index(x) if x in ADDUCT_VOCAB else len(ADDUCT_VOCAB) for x in adduct], np.int64)
    l = np.array([LIB_VOCAB.index(x) if x in LIB_VOCAB else len(LIB_VOCAB) for x in lib], np.int64)
    i = np.array([instr_class(x) for x in instr], np.int64)
    m = np.array([1 if x == 'positive' else 0 for x in mode], np.int64)
    ce = np.asarray(ce_mean, np.float32); nan = ~np.isfinite(ce)
    cef = np.stack([np.where(nan, 0.0, np.log1p(np.clip(ce, 0, 500)) / 6.0), nan.astype(np.float32), (np.asarray(ce_n) > 1).astype(np.float32)], 1).astype(np.float32)
    shift = np.array([ADDUCT_SHIFT.get(x, PROTON) for x in adduct], np.float64)
    return dict(adduct=a, lib=l, instr=i, mode=m, cef=cef, prec=np.asarray(prec, np.float32), shift=shift.astype(np.float32))


class Fourier(nn.Module):
    def __init__(self, n=64, wl_min=0.005, wl_max=3000.0):
        super().__init__()
        wl = torch.exp(torch.linspace(math.log(wl_min), math.log(wl_max), n))
        self.register_buffer('freq', 2 * math.pi / wl)

    def forward(self, x):  # x [...], returns [..., 2n]
        v = x.unsqueeze(-1) * self.freq
        return torch.cat([torch.sin(v), torch.cos(v)], -1)


class SpecModel(nn.Module):
    def __init__(self, d=512, layers=6, heads=8, ff=2048, drop=0.1, n_bits=4096, n_maccs=167, n_cnt=2048, emb=512):
        super().__init__()
        self.four = Fourier(64)
        self.peak_in = nn.Sequential(nn.Linear(128 * 2 + 4, d), nn.GELU(), nn.Linear(d, d))
        self.adduct = nn.Embedding(len(ADDUCT_VOCAB) + 1, d); self.lib = nn.Embedding(len(LIB_VOCAB) + 1, d)
        self.instr = nn.Embedding(len(INSTR_CLASSES), d); self.mode = nn.Embedding(2, d)
        self.glob_in = nn.Sequential(nn.Linear(128 + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head_bits = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Linear(2048, n_bits))
        self.head_maccs = nn.Linear(d, n_maccs)
        self.head_cnt = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Linear(2048, n_cnt))
        self.head_el = nn.Linear(d, len(ELEMENTS))
        self.head_emb = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, mz, it, npk, meta):
        h, _ = self.encode(mz, it, npk, meta)
        h = h[:, 0]
        return dict(bits=self.head_bits(h), maccs=self.head_maccs(h), cnt=self.head_cnt(h), el=self.head_el(h), emb=F.normalize(self.head_emb(h), dim=-1))

    def encode(self, mz, it, npk, meta):
        """-> token states [B, 1+K, d] (index 0 = CLS, normalised) and key-padding mask [B, 1+K] (True = pad)."""
        B, Kk = mz.shape
        mask = torch.arange(Kk, device=mz.device)[None] >= npk[:, None]   # True = pad
        prec = meta['prec'][:, None]
        loss = prec - mz
        rel = torch.where(prec > 0, mz / prec.clamp(min=1.0), torch.zeros_like(mz))
        pf = torch.cat([self.four(mz), self.four(loss), torch.log1p(it * 100)[..., None] / 4.6, torch.sqrt(it)[..., None], rel[..., None],
                        (torch.abs(loss) < 0.5).float()[..., None]], -1)
        x = self.peak_in(pf)
        g = self.glob_in(torch.cat([self.four(meta['prec']), meta['cef'], torch.log1p(meta['prec'])[:, None] / 7.0], -1))
        g = g + self.adduct(meta['adduct']) + self.lib(meta['lib']) + self.instr(meta['instr']) + self.mode(meta['mode'])
        x = torch.cat([self.cls.expand(B, -1, -1) + g[:, None], x], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mz.device), mask], 1)
        return self.norm(self.enc(x, src_key_padding_mask=mask)), mask


class MolEncoder(nn.Module):
    def __init__(self, d=1024, emb=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048 + 167, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb))

    def forward(self, cnt, maccs):   # cnt [B,2048] uint8-ish floats, maccs [B,167]
        return F.normalize(self.net(torch.cat([torch.log1p(cnt.float()), maccs.float()], -1)), dim=-1)


def unpack_bits(packed, n):
    return np.unpackbits(packed, axis=-1)[..., :n]


def fp_loglik(p_bits, y_bits, eps=1e-4):
    """p [B,n] probabilities, y [C,n] 0/1 -> [B,C] log-likelihood of each candidate under each prediction."""
    lp = torch.log(p_bits.clamp(eps, 1 - eps)); lq = torch.log((1 - p_bits).clamp(eps, 1 - eps))
    return (lp - lq) @ y_bits.T + lq.sum(-1, keepdim=True)
