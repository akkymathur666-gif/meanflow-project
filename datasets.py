"""2-D toy datasets (numpy only)."""
import numpy as np


def _moons(n, rng, noise=0.05):
    n1 = n // 2
    th1 = rng.uniform(0, np.pi, n1)
    th2 = rng.uniform(0, np.pi, n - n1)
    a = np.stack([np.cos(th1), np.sin(th1)], 1)
    b = np.stack([1 - np.cos(th2), 0.5 - np.sin(th2)], 1)
    x = np.concatenate([a, b], 0) + noise * rng.normal(size=(n, 2))
    return (x - np.array([0.5, 0.25])) * 2.0


def _gaussians8(n, rng, radius=2.0, std=0.15):
    ang = rng.integers(0, 8, n) * (2 * np.pi / 8)
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    return centers + std * rng.normal(size=(n, 2))


def _spiral(n, rng, noise=0.03):
    u = rng.uniform(0, 1, n)
    th = 3 * np.pi * np.sqrt(u)
    rad = 2.0 * th / (3 * np.pi)
    x = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
    return x + noise * rng.normal(size=(n, 2))


_FUNCS = {"moons": _moons, "8gaussians": _gaussians8, "spiral": _spiral}


def sample_dataset(name, n, seed=0):
    rng = np.random.default_rng(seed)
    return _FUNCS[name](n, rng).astype(np.float32)


DATASETS = list(_FUNCS)
