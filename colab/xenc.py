"""Spectrum x candidate cross-encoder (own model).
Spectrum side: the pretrained SpecModel transformer (token states). Candidate side: one global token (count
fingerprint), atom tokens and cleavage-fragment tokens (Fourier features of the fragment ion m/z and of the
corresponding neutral loss from the precursor). A transformer decoder lets candidate tokens attend to the peak
tokens; the global token's output gives the pair score. Inputs for the candidates are shaped [B, G, ...] so that
DataParallel can split spectra and their candidate groups together.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from fpmodel import SpecModel, Fourier

PROTON = 1.007276


class XEnc(nn.Module):
    def __init__(self, spec: SpecModel, spec_d=512, d=256, layers=3, heads=8, drop=0.1):
        super().__init__()
        self.spec = spec
        self.proj = nn.Sequential(nn.Linear(spec_d, d), nn.LayerNorm(d))
        self.el = nn.Embedding(14, d); self.deg = nn.Embedding(8, d); self.hs = nn.Embedding(6, d); self.aro = nn.Embedding(2, d); self.ring = nn.Embedding(2, d)
        self.four = Fourier(64)
        self.frag_in = nn.Sequential(nn.Linear(128 * 4 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.glob_in = nn.Sequential(nn.Linear(2048, d), nn.GELU(), nn.Linear(d, d))
        self.type = nn.Embedding(3, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, drop, activation='gelu', batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, mz, it, npk, adduct, lib, instr, mode, cef, prec, atoms, frags, nn_, cnt):
        meta = dict(adduct=adduct, lib=lib, instr=instr, mode=mode, cef=cef, prec=prec)
        h, smask = self.spec.encode(mz, it, npk, meta)                # [B, L, 512], [B, L]
        mem = self.proj(h)
        B, G, A = atoms.shape[:3]; Fm = frags.shape[2]; L, d = mem.shape[1], mem.shape[2]
        mem = mem[:, None].expand(B, G, L, d).reshape(B * G, L, d); smask = smask[:, None].expand(B, G, L).reshape(B * G, L)
        atoms = atoms.long()
        atok = self.el(atoms[..., 0].clamp(0, 13)) + self.deg(atoms[..., 1].clamp(0, 7)) + self.hs(atoms[..., 2].clamp(0, 5)) + self.aro(atoms[..., 3].clamp(0, 1)) + self.ring(atoms[..., 4].clamp(0, 1))
        shift = torch.where(mode == 1, torch.full_like(prec, PROTON), torch.full_like(prec, -PROTON))[:, None, None]   # [B,1,1]
        P = prec[:, None, None]
        fa = frags[..., 0]; fb = frags[..., 1]
        ia = fa + shift; ib = fb + shift
        rel = torch.stack([ia / P.clamp(min=1), ib / P.clamp(min=1)], -1)
        ftok = self.frag_in(torch.cat([self.four(ia), self.four(ib), self.four(P - ia), self.four(P - ib), rel], -1))
        gtok = self.glob_in(torch.log1p(cnt.float()) / 3.0)[:, :, None]   # [B, G, 1, d]
        toks = torch.cat([gtok + self.type.weight[0], atok + self.type.weight[1], ftok + self.type.weight[2]], 2).reshape(B * G, 1 + A + Fm, d)
        na = nn_[..., 0].long().reshape(B * G); nf = nn_[..., 1].long().reshape(B * G)
        ar = torch.arange(A, device=mz.device)[None]; fr = torch.arange(Fm, device=mz.device)[None]
        tmask = torch.cat([torch.zeros(B * G, 1, dtype=torch.bool, device=mz.device), ar >= na[:, None], fr >= nf[:, None]], 1)
        out = self.dec(toks, mem, tgt_key_padding_mask=tmask, memory_key_padding_mask=smask)
        return self.head(self.norm(out[:, 0])).view(B, G)
