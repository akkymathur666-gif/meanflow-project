"""
2-D toy experiments: flow matching (FM) vs MeanFlow (MF), plus ablations.

  python run_toy.py --quick                  # ~1 minute smoke test
  python run_toy.py --iters 10000            # main experiment (all 3 datasets) + ablations
Outputs (in --out): samples_<dataset>.png, results.json, results.md
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import DATASETS, sample_dataset
from meanflow import (
    ToyMeanFlowNet,
    ema_update,
    flow_matching_loss,
    meanflow_loss,
    sample_fm,
    sample_meanflow,
    sliced_wasserstein,
)

N_EVAL = 5000


def train_model(kind, ds, iters, bs, lr, seed, device, **loss_kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ToyMeanFlowNet().to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    data = torch.from_numpy(sample_dataset(ds, 200_000, seed=1234)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    t0 = time.time()
    for it in range(iters):
        x = data[torch.randint(0, data.shape[0], (bs,), device=device)]
        if kind == "fm":
            loss = flow_matching_loss(model, x)
        else:
            loss, _ = meanflow_loss(model, x, **loss_kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_update(ema, model, 0.999)
        if (it + 1) % max(1, iters // 5) == 0:
            print(f"  [{kind}/{ds}] it {it+1}/{iters} loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    return ema, time.time() - t0


@torch.no_grad()
def evaluate(model, kind, ds, steps_list, device):
    ref = sample_dataset(ds, N_EVAL, seed=999)
    out, samples = {}, {}
    for s in steps_list:
        torch.manual_seed(0)
        fn = sample_fm if kind == "fm" else sample_meanflow
        x = fn(model, (N_EVAL, 2), s, device).cpu().numpy()
        samples[s] = x
        out[s] = sliced_wasserstein(x, ref)
    return out, samples


def scatter(ax, pts, title, lim):
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.35)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation_dataset", default="8gaussians")
    ap.add_argument("--ablation_seeds", type=int, default=3)
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--out", default="results_toy")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.ablation_seeds = 300, 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    print("device:", device)
    results = {"args": vars(args), "main": {}, "ablation": {}}
    fm_steps, mf_steps = [1, 2, 4, 8, 32, 128], [1, 2, 4, 8, 32]

    # ---------------- main comparison ----------------
    for ds in args.datasets:
        ref = sample_dataset(ds, N_EVAL, seed=999)
        floor = sliced_wasserstein(sample_dataset(ds, N_EVAL, seed=998), ref)
        lim = float(np.abs(ref).max()) + 0.5
        fm, t_fm = train_model("fm", ds, args.iters, args.bs, args.lr, args.seed, device)
        mf, t_mf = train_model("mf", ds, args.iters, args.bs, args.lr, args.seed, device)
        fm_sw, fm_s = evaluate(fm, "fm", ds, fm_steps, device)
        mf_sw, mf_s = evaluate(mf, "mf", ds, mf_steps, device)
        results["main"][ds] = {
            "noise_floor_sw2": floor,
            "fm_sw2": fm_sw,
            "mf_sw2": mf_sw,
            "train_seconds": {"fm": t_fm, "mf": t_mf},
        }
        fig, axs = plt.subplots(1, 6, figsize=(15, 2.8))
        scatter(axs[0], ref, "data", lim)
        scatter(axs[1], fm_s[1], f"FM 1-step  SW2={fm_sw[1]:.3f}", lim)
        scatter(axs[2], fm_s[8], f"FM 8-step  SW2={fm_sw[8]:.3f}", lim)
        scatter(axs[3], fm_s[128], f"FM 128-step SW2={fm_sw[128]:.3f}", lim)
        scatter(axs[4], mf_s[1], f"MeanFlow 1-step SW2={mf_sw[1]:.3f}", lim)
        scatter(axs[5], mf_s[2], f"MeanFlow 2-step SW2={mf_sw[2]:.3f}", lim)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"samples_{ds}.png"), dpi=150)
        plt.close(fig)

    # ---------------- ablations ----------------
    if not args.skip_ablation:
        ds = args.ablation_dataset
        configs = {
            "default (p_neq=0.25, jvp, adaptive)": dict(),
            "p_neq=0.0 (pure FM targets)": dict(p_neq=0.0),
            "p_neq=0.5": dict(p_neq=0.5),
            "p_neq=1.0": dict(p_neq=1.0),
            "finite-difference instead of JVP": dict(dt_mode="fd"),
            "no adaptive loss weighting": dict(adaptive_p=0.0),
        }
        for name, kw in configs.items():
            runs = {1: [], 2: []}
            for sd in range(args.ablation_seeds):
                m, _ = train_model("mf", ds, args.iters, args.bs, args.lr, 100 + sd, device, **kw)
                sw, _ = evaluate(m, "mf", ds, [1, 2], device)
                for k in runs:
                    runs[k].append(sw[k])
            results["ablation"][name] = {
                f"sw2_{k}step_mean": float(np.mean(v)) for k, v in runs.items()
            } | {f"sw2_{k}step_std": float(np.std(v)) for k, v in runs.items()}
            print("ablation:", name, results["ablation"][name])

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    # ---------------- markdown tables ----------------
    L = ["## Main comparison (sliced Wasserstein-2, lower is better)\n"]
    L.append("| dataset | noise floor | FM 1 | FM 8 | FM 128 | MF 1 | MF 2 | MF 8 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ds, r in results["main"].items():
        f, m = r["fm_sw2"], r["mf_sw2"]
        L.append(
            f"| {ds} | {r['noise_floor_sw2']:.3f} | {f[1]:.3f} | {f[8]:.3f} | {f[128]:.3f} "
            f"| {m[1]:.3f} | {m[2]:.3f} | {m[8]:.3f} |"
        )
    if results["ablation"]:
        L.append(f"\n## Ablations on {args.ablation_dataset} (mean ± std over {args.ablation_seeds} seeds)\n")
        L.append("| config | SW2 1-step | SW2 2-step |")
        L.append("|---|---|---|")
        for n, r in results["ablation"].items():
            L.append(
                f"| {n} | {r['sw2_1step_mean']:.3f} ± {r['sw2_1step_std']:.3f} "
                f"| {r['sw2_2step_mean']:.3f} ± {r['sw2_2step_std']:.3f} |"
            )
    open(os.path.join(args.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
