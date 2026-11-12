"""
MNIST / CIFAR-10: flow matching vs MeanFlow (1-step) on a small U-Net.

  python train_images.py --dataset mnist   --iters 20000 --models fm mf
  python train_images.py --dataset cifar10 --iters 60000 --models fm mf
  python train_images.py --dataset mnist   --eval_only         # reload checkpoints, re-evaluate

Everything is fp32 (torch.func.jvp + AMP on a T4 is asking for trouble).
If torch.func.jvp complains that an op has no forward-mode AD formula, rerun with
--dt_mode fd (finite-difference fallback) and report that in the failure-cases section.
FID needs `pip install torchmetrics torch-fidelity` and internet (Inception weights download).
"""
import argparse
import copy
import os
import time

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from meanflow import (
    MeanFlowUNet,
    ema_update,
    flow_matching_loss,
    meanflow_loss,
    sample_fm,
    sample_meanflow,
)


def get_data(name, root, bs):
    if name == "mnist":
        tf = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        ds = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
        ch = 1
    else:
        tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
        ds = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
        ch = 3
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    return ds, dl, ch


def infinite(dl):
    while True:
        for x, _ in dl:
            yield x


def to01(x):
    return ((x.clamp(-1, 1) + 1) / 2)


@torch.no_grad()
def fid_score(sample_fn, real_ds, n, device, ch):
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    loader = DataLoader(real_ds, batch_size=250, shuffle=False, num_workers=2)
    seen = 0
    for x, _ in loader:
        x = to01(x.to(device))
        fid.update(x.repeat(1, 3 // ch, 1, 1), real=True)
        seen += x.shape[0]
        if seen >= n:
            break
    done = 0
    while done < n:
        b = min(250, n - done)
        x = to01(sample_fn(b))
        fid.update(x.repeat(1, 3 // ch, 1, 1), real=False)
        done += b
    return fid.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--models", nargs="+", default=["fm", "mf"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--p_neq", type=float, default=0.25)
    ap.add_argument("--mu", type=float, default=-0.4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--adaptive_p", type=float, default=1.0)
    ap.add_argument("--dt_mode", choices=["jvp", "fd"], default="jvp")
    ap.add_argument("--fid_n", type=int, default=10000)
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results_images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = os.path.join(args.out, args.dataset)
    os.makedirs(out, exist_ok=True)
    ds, dl, ch = get_data(args.dataset, args.root, args.bs)
    size = ds[0][0].shape[-1]
    shape = lambda b: (b, ch, size, size)
    report = []

    for kind in args.models:
        ckpt = os.path.join(out, f"{kind}_ema.pt")
        model = MeanFlowUNet(in_ch=ch, base=args.base).to(device)
        ema = copy.deepcopy(model).requires_grad_(False)
        if not args.eval_only:
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            stream, t0 = infinite(dl), time.time()
            for it in range(args.iters):
                x = next(stream).to(device, non_blocking=True)
                if kind == "fm":
                    loss = flow_matching_loss(model, x)
                else:
                    loss, _ = meanflow_loss(
                        model, x, p_neq=args.p_neq, mu=args.mu, sigma=args.sigma,
                        adaptive_p=args.adaptive_p, dt_mode=args.dt_mode,
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema_update(ema, model, 0.9995)
                if (it + 1) % 500 == 0:
                    print(f"[{kind}] it {it+1}/{args.iters} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
                if (it + 1) % 5000 == 0 or it + 1 == args.iters:
                    torch.save(ema.state_dict(), ckpt)
                    s = sample_meanflow(ema, shape(64), 1, device) if kind == "mf" else sample_fm(ema, shape(64), 20, device)
                    save_image(to01(s), os.path.join(out, f"{kind}_progress.png"), nrow=8)
            print(f"[{kind}] training time {time.time()-t0:.0f}s")
        else:
            ema.load_state_dict(torch.load(ckpt, map_location=device))
        ema.eval()

        configs = [("mf", n) for n in (1, 2, 4)] if kind == "mf" else [("fm", n) for n in (1, 2, 10, 50, 100)]
        for k, n in configs:
            fn = (lambda b, n=n: sample_meanflow(ema, shape(b), n, device)) if k == "mf" else \
                 (lambda b, n=n: sample_fm(ema, shape(b), n, device))
            torch.manual_seed(1)
            save_image(to01(fn(64)), os.path.join(out, f"{k}_{n}step.png"), nrow=8)
            fid = None
            if not args.no_fid:
                try:
                    fid = fid_score(fn, ds, args.fid_n, device, ch)
                except Exception as e:  # missing torchmetrics / no internet
                    print("FID skipped:", repr(e))
            report.append((k, n, fid))
            print(f"{k} {n}-step  FID({args.fid_n}) = {fid}")

    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(f"| model | steps | FID ({args.fid_n} samples, {args.dataset}) |\n|---|---|---|\n")
        for k, n, fid in report:
            f.write(f"| {k} | {n} | {'n/a' if fid is None else f'{fid:.2f}'} |\n")
    print(open(os.path.join(out, "results.md")).read())


if __name__ == "__main__":
    main()
