"""
MeanFlow (Geng, Deng, Bai, Kolter, He, 2025; arXiv:2505.13447)
Minimal from-scratch PyTorch implementation of the core contribution:

  * average-velocity network  u_theta(z, r, t)
  * MeanFlow identity target  u_tgt = v - (t - r) * d/dt u_theta(z_t, r, t)
      where d/dt u = (dz/dt) . grad_z u + d_t u   -> ONE Jacobian-vector product
      with tangent (v, 0, 1)
  * 1-step / few-step sampler and a flow-matching baseline (same network, r = t)

Conventions (same as the paper):  z_t = (1 - t) x + t * eps,   v = eps - x,
t = 1 is noise, t = 0 is data.  One-step sample:  x = eps - u_theta(eps, 0, 1).

NOTE: the paper's exact hyper-parameters (t/r sampling, fraction of r != t,
loss-weighting constants) must be taken from the paper's tables/appendix; the
defaults below are my own reasonable choices and are exposed as arguments.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp


def _bshape(v, ref):
    """(B,) -> (B,1,...,1) so it broadcasts against ref."""
    return v.view(-1, *([1] * (ref.dim() - 1)))


# --------------------------------------------------------------------------
# Embeddings / networks
# --------------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of a (B,) tensor of times in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = (t * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ToyMeanFlowNet(nn.Module):
    """Residual MLP u_theta(z, r, t) for low-dimensional data. Conditioned on (t, t - r)."""

    def __init__(self, dim=2, hidden=256, depth=4, temb=64):
        super().__init__()
        self.temb = temb
        self.inp = nn.Linear(dim + 2 * temb, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, z, r, t):
        c = torch.cat(
            [z, timestep_embedding(t, self.temb), timestep_embedding(t - r, self.temb)], dim=-1
        )
        h = self.inp(c)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cemb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(cemb, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class MeanFlowUNet(nn.Module):
    """
    Small conv U-Net for 28x28 / 32x32 images. Only ops with forward-mode AD
    support are used (conv, conv-transpose, GroupNorm, SiLU, Linear, concat), so
    torch.func.jvp works. Down/up-sampling is done with strided conv / conv-transpose.
    """

    def __init__(self, in_ch=1, base=64, mults=(1, 2, 2), temb=128):
        super().__init__()
        self.temb = temb
        cemb = base * 4
        self.emb = nn.Sequential(nn.Linear(2 * temb, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down_blocks, self.downs = nn.ModuleList(), nn.ModuleList()
        c = base
        for i, ch in enumerate(chs):
            self.down_blocks.append(ResBlock(c, ch, cemb))
            c = ch
            if i < len(chs) - 1:
                self.downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid = ResBlock(c, c, cemb)
        self.up_blocks, self.ups = nn.ModuleList(), nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            self.up_blocks.append(ResBlock(c + ch, ch, cemb))
            c = ch
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(c, c, 2, stride=2))
        self.outn = nn.GroupNorm(8, c)
        self.outc = nn.Conv2d(c, in_ch, 3, padding=1)

    def forward(self, z, r, t):
        emb = self.emb(
            torch.cat([timestep_embedding(t, self.temb), timestep_embedding(t - r, self.temb)], -1)
        )
        h = self.inc(z)
        skips = []
        for i, blk in enumerate(self.down_blocks):
            h = blk(h, emb)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)
        h = self.mid(h, emb)
        for j, blk in enumerate(self.up_blocks):
            h = blk(torch.cat([h, skips.pop()], 1), emb)
            if j < len(self.ups):
                h = self.ups[j](h)
        return self.outc(F.silu(self.outn(h)))


# --------------------------------------------------------------------------
# Training losses
# --------------------------------------------------------------------------
def sample_r_t(B, device, p_neq=0.25, mu=-0.4, sigma=1.0):
    """
    Sample (r, t) with r <= t from a logit-normal; with prob (1 - p_neq) set r = t
    (those samples reduce to plain flow matching).
    """
    a = torch.sigmoid(torch.randn(B, device=device) * sigma + mu)
    b = torch.sigmoid(torch.randn(B, device=device) * sigma + mu)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    keep_diff = torch.rand(B, device=device) < p_neq
    r = torch.where(keep_diff, r, t)
    return r, t


def flow_matching_loss(model, x):
    """Standard flow matching baseline: regress v = eps - x using u_theta(z_t, t, t)."""
    B = x.shape[0]
    eps = torch.randn_like(x)
    t = torch.rand(B, device=x.device)
    z = (1 - _bshape(t, x)) * x + _bshape(t, x) * eps
    v = eps - x
    pred = model(z, t, t)
    return (pred - v).pow(2).flatten(1).mean(1).mean()


def meanflow_loss(
    model,
    x,
    p_neq=0.25,
    mu=-0.4,
    sigma=1.0,
    adaptive_p=1.0,
    adaptive_c=1e-3,
    dt_mode="jvp",
    fd_eps=1e-3,
):
    """
    MeanFlow loss.

      u_tgt = v - (t - r) * ( v . d_z u + d_t u )          (stop-gradient)
      loss  = || u_theta(z_t, r, t) - u_tgt ||^2  (optionally adaptively weighted)

    dt_mode = "jvp": exact total derivative via one torch.func.jvp call
              "fd" : central finite difference along (v, 0, 1)   (ablation)
    adaptive_p = 0 disables the adaptive loss weighting w = 1 / (loss + c)^p.
    """
    B = x.shape[0]
    eps = torch.randn_like(x)
    r, t = sample_r_t(B, x.device, p_neq, mu, sigma)
    tb, rb = _bshape(t, x), _bshape(r, x)
    z = (1 - tb) * x + tb * eps
    v = eps - x

    if dt_mode == "jvp":
        u, dudt = jvp(model, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    elif dt_mode == "fd":
        u = model(z, r, t)
        with torch.no_grad():
            up = model(z + fd_eps * v, r, t + fd_eps)
            um = model(z - fd_eps * v, r, t - fd_eps)
            dudt = (up - um) / (2 * fd_eps)
    else:
        raise ValueError(dt_mode)

    u_tgt = (v - (tb - rb) * dudt).detach()  # stop-gradient on the target
    sq = (u - u_tgt).pow(2).flatten(1).mean(1)  # per-sample squared error
    if adaptive_p > 0:
        w = 1.0 / (sq.detach() + adaptive_c).pow(adaptive_p)
        loss = (w * sq).mean()
    else:
        loss = sq.mean()
    return loss, {"raw_mse": sq.mean().item()}


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------
@torch.no_grad()
def sample_meanflow(model, shape, n_steps=1, device="cpu"):
    """z_r = z_t - (t - r) * u_theta(z_t, r, t); n_steps = 1 is the one-step sampler."""
    z = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i].expand(shape[0])
        r = ts[i + 1].expand(shape[0])
        z = z - _bshape(t - r, z) * model(z, r, t)
    return z


@torch.no_grad()
def sample_fm(model, shape, n_steps=50, device="cpu"):
    """Euler sampler for the flow-matching baseline (instantaneous velocity = u(z, t, t))."""
    z = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t = ts[i].expand(shape[0])
        dt = ts[i] - ts[i + 1]
        z = z - dt * model(z, t, t)
    return z


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)


def sliced_wasserstein(a, b, n_proj=256, seed=0):
    """Sliced Wasserstein-2 distance between two equal-size point clouds (numpy)."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_proj, a.shape[1]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pa, pb = np.sort(a @ d.T, axis=0), np.sort(b @ d.T, axis=0)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))
